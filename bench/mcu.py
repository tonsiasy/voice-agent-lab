"""最小 MCU（Multipoint Control Unit，多点控制单元）：解码 → 合成 → 再编码。

与 SFU（Selective Forwarding Unit，选择性转发单元）的根本差别在这里一目了然：
SFU 只是把收到的 RTP 包转发出去，从不解码；MCU 必须把每一路解码成像素/PCM，
合成一路新的，再编码一次。**成本从 O(转发) 变成 O(N×解码 + 1×编码)。**

三条设计要点（都是写的时候会自然撞上的约束）：

1. **输出时钟必须独立于输入**。合成循环按自己的节拍跑，不等任何一路。
   若改成「等所有输入都到齐再合成」，最慢的那一路会拖垮全局。
2. **latest-wins,不排队**。每路只保留最新一帧;某路卡住时它的格子静止，
   其余格子照常刷新——这正是 MCU 的降级形态。
3. **视频是空间拼接,音频是数值叠加**。视频把 N 路摆进网格;音频把 N 路
   波形相加(要防削顶)。两者算法完全不同,这是 MCU 里最容易被忽略的一半。
4. **音视频必须分开跑,不能共用一个事件循环**——本实验实测出来的教训。
   合成一帧 1280x720 要约 32ms,而混音每 10ms 就得出一帧;
   合成把循环占死,混音根本轮不上,每秒丢 82 帧音频。
   降到 640x360(合成 7.8ms)后丢帧掉到 3 帧/秒,**对照实验证实了因果**。
   生产级 MCU 把音频与视频放在不同线程/进程正是为此
   (LiveKit 的 Egress 独立成服务也有这层原因)。本文件保持单循环,
   是为了让这个约束暴露出来而不是藏起来。

用法:
    PYTHONPATH=bench uv run python bench/mcu.py [room] [输出mp4]
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time

import numpy as np
from livekit import rtc

from lkutil import URL, token
from mcu_layout import cell_rects, scale_into

OUT_W = int(os.environ.get("MCU_OUT_W", 1280))
OUT_H = int(os.environ.get("MCU_OUT_H", 720))
OUT_FPS = int(os.environ.get("MCU_OUT_FPS", 10))
SR, CHANNELS = 48000, 1
AUDIO_FRAME_MS = 10
AUDIO_QUEUE_MAX = 5           # 每路音频最多缓 50ms;再多就是在积压延迟
BACKGROUND = 24               # 画布底色(深灰),空格子留白用


class VideoSlot:
    """一路视频输入。只保留最新一帧——见模块注释第 2 条。"""

    def __init__(self, identity: str, track: rtc.RemoteVideoTrack) -> None:
        self.identity = identity
        self.frame: np.ndarray | None = None
        self.decoded = 0
        self.reported = 0        # 基准存在槽位上:轨重订阅时槽位重建,基准随之归零
        self._task = asyncio.create_task(self._run(track))

    async def _run(self, track: rtc.RemoteVideoTrack) -> None:
        # format=RGBA:让 SDK 直接给 RGBA,省掉我们自己做 I420 转换
        stream = rtc.VideoStream(track, format=rtc.VideoBufferType.RGBA)
        try:
            async for ev in stream:
                f = ev.frame
                self.frame = np.frombuffer(f.data, dtype=np.uint8).reshape(
                    f.height, f.width, 4
                )
                self.decoded += 1
        except Exception:
            pass

    def close(self) -> None:
        self._task.cancel()


class AudioSlot:
    """一路音频输入。有界队列:宁可丢旧帧,也不让延迟无限增长。"""

    def __init__(self, identity: str, track: rtc.RemoteAudioTrack) -> None:
        self.identity = identity
        self.queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=AUDIO_QUEUE_MAX)
        self.dropped = 0
        self._task = asyncio.create_task(self._run(track))

    async def _run(self, track: rtc.RemoteAudioTrack) -> None:
        stream = rtc.AudioStream(
            track, sample_rate=SR, num_channels=CHANNELS, frame_size_ms=AUDIO_FRAME_MS
        )
        try:
            async for ev in stream:
                pcm = np.frombuffer(ev.frame.data, dtype=np.int16)
                if self.queue.full():
                    self.queue.get_nowait()
                    self.dropped += 1
                self.queue.put_nowait(pcm)
        except Exception:
            pass

    def take(self, n: int) -> np.ndarray | None:
        """排到多帧说明混音落后了:丢旧取新,与视频的 latest-wins 保持一致。

        音频丢帧一定有可闻的瑕疵,取舍是「短暂爆音」还是「持续增长的延迟」——
        实时会议里必须选前者。生产级 MCU 在这里会放抖动缓冲做平滑。
        """
        pcm = None
        while True:
            try:
                candidate = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if pcm is not None:
                self.dropped += 1
            pcm = candidate
        return pcm if pcm is not None and len(pcm) == n else None

    def close(self) -> None:
        self._task.cancel()


class MiniMCU:
    def __init__(self, room: rtc.Room, mp4_path: str | None) -> None:
        self.room = room
        self.video_slots: dict[str, VideoSlot] = {}
        self.audio_slots: dict[str, AudioSlot] = {}
        self.video_source = rtc.VideoSource(OUT_W, OUT_H)
        self.audio_source = rtc.AudioSource(SR, CHANNELS)
        self.composite_ms = 0.0
        self.composited = 0
        self._recorder = self._start_recorder(mp4_path) if mp4_path else None

    @staticmethod
    def _start_recorder(path: str) -> subprocess.Popen:
        """录制合流——这正是生产环境里 Egress 干的活,只是它还管布局模板与切片。"""
        # 分片 MP4(frag_keyframe+empty_moov):不依赖进程正常退出去写 moov 原子。
        # 录制类进程随时可能被杀,普通 MP4 一旦没收尾整个文件都打不开。
        return subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "rgba",
             "-s", f"{OUT_W}x{OUT_H}", "-r", str(OUT_FPS), "-i", "-",
             "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
             "-movflags", "frag_keyframe+empty_moov+default_base_moof", path],
            stdin=subprocess.PIPE,
        )

    def on_track(self, track, participant) -> None:
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            self.video_slots[participant.identity] = VideoSlot(participant.identity, track)
        elif track.kind == rtc.TrackKind.KIND_AUDIO:
            self.audio_slots[participant.identity] = AudioSlot(participant.identity, track)
        print(f"[+] {participant.identity} {track.kind}  "
              f"视频 {len(self.video_slots)} 路 / 音频 {len(self.audio_slots)} 路", flush=True)

    def on_track_gone(self, participant) -> None:
        for slots in (self.video_slots, self.audio_slots):
            slot = slots.pop(participant.identity, None)
            if slot:
                slot.close()

    async def publish(self) -> None:
        await self.room.local_participant.publish_track(
            rtc.LocalVideoTrack.create_video_track("mcu-composite", self.video_source),
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA),
        )
        await self.room.local_participant.publish_track(
            rtc.LocalAudioTrack.create_audio_track("mcu-mix", self.audio_source),
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
        )

    async def compose_video(self) -> None:
        """固定节拍合成——不等输入,见模块注释第 1 条。"""
        canvas = np.full((OUT_H, OUT_W, 4), BACKGROUND, dtype=np.uint8)
        canvas[..., 3] = 255
        interval = 1 / OUT_FPS
        next_at = time.perf_counter()
        while True:
            next_at += interval
            # 按身份排序:格子位置稳定,不随订阅到达顺序变化
            slots = [s for _, s in sorted(self.video_slots.items()) if s.frame is not None]
            t0 = time.perf_counter()
            canvas[:] = BACKGROUND
            canvas[..., 3] = 255
            for slot, (top, left, ch, cw) in zip(slots, cell_rects(max(len(slots), 1), OUT_W, OUT_H)):
                # 取最新帧;这一路卡住则它的格子静止,其余格子照常刷新
                scale_into(canvas, slot.frame, top, left, ch, cw)
            self.composite_ms += (time.perf_counter() - t0) * 1000
            self.composited += 1

            raw = canvas.tobytes()
            self.video_source.capture_frame(
                rtc.VideoFrame(OUT_W, OUT_H, rtc.VideoBufferType.RGBA, raw)
            )
            if self._recorder and self._recorder.stdin:
                self._recorder.stdin.write(raw)

            await asyncio.sleep(max(0.0, next_at - time.perf_counter()))

    async def mix_audio(self) -> None:
        """波形相加。int32 累加后再限幅——直接 int16 相加会静默溢出翻转。"""
        n = SR * AUDIO_FRAME_MS // 1000
        interval = AUDIO_FRAME_MS / 1000
        next_at = time.perf_counter()
        while True:
            next_at += interval
            acc = np.zeros(n, dtype=np.int32)
            for slot in list(self.audio_slots.values()):
                pcm = slot.take(n)
                if pcm is not None:
                    acc += pcm
            mixed = np.clip(acc, -32768, 32767).astype(np.int16)
            await self.audio_source.capture_frame(
                rtc.AudioFrame(mixed.tobytes(), SR, CHANNELS, n)
            )
            await asyncio.sleep(max(0.0, next_at - time.perf_counter()))

    async def report(self) -> None:
        """每秒一行:解码帧数、合成耗时、本进程 CPU 占用。"""
        last_cpu, last_wall = time.process_time(), time.perf_counter()
        while True:
            await asyncio.sleep(1.0)
            cpu, wall = time.process_time(), time.perf_counter()
            util = (cpu - last_cpu) / max(wall - last_wall, 1e-6) * 100
            last_cpu, last_wall = cpu, wall

            fps_in = []
            for ident, slot in sorted(self.video_slots.items()):
                fps_in.append(f"{ident}:{slot.decoded - slot.reported}")
                slot.reported = slot.decoded
            avg_ms = self.composite_ms / max(self.composited, 1)
            self.composite_ms = self.composited = 0
            drops = sum(s.dropped for s in self.audio_slots.values())
            print(f"入 [{' '.join(fps_in) or '-'}]  合成 {avg_ms:.2f}ms/帧  "
                  f"音频 {len(self.audio_slots)} 路(丢 {drops})  CPU {util:.0f}%", flush=True)

    def close(self) -> None:
        for slots in (self.video_slots, self.audio_slots):
            for slot in slots.values():
                slot.close()
        if self._recorder and self._recorder.stdin:
            self._recorder.stdin.close()
            self._recorder.wait(timeout=10)


async def main() -> None:
    room_name = sys.argv[1] if len(sys.argv) > 1 else "day8"
    mp4_path = sys.argv[2] if len(sys.argv) > 2 else None

    room = rtc.Room()
    mcu = MiniMCU(room, mp4_path)

    @room.on("track_subscribed")
    def _on(track, pub, participant):
        mcu.on_track(track, participant)

    @room.on("participant_disconnected")
    def _off(participant):
        mcu.on_track_gone(participant)

    await room.connect(URL, token("mcu", room_name))
    print(f"MCU 已入房 {room_name}；输出 {OUT_W}x{OUT_H}@{OUT_FPS}"
          + (f"，录制 → {mp4_path}" if mp4_path else ""), flush=True)
    await mcu.publish()

    try:
        await asyncio.gather(mcu.compose_video(), mcu.mix_audio(), mcu.report())
    finally:
        mcu.close()
        await room.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
