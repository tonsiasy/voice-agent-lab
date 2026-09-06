"""Day 8 MCU 实验的信号源：发布可辨识的视频 + 音频。

每路用不同底色、画面正中一个大数字、以及不同音高的正弦音。
这样 MCU 合成得对不对——视频看一眼(四格四色四数字)、音频听一耳(和弦)——立刻能判断。

用法: PYTHONPATH=bench uv run python bench/mcu_source.py <序号 0-3> [room]
"""
from __future__ import annotations

import asyncio
import sys

import numpy as np
from livekit import rtc
from PIL import Image, ImageDraw, ImageFont

from lkutil import URL, token

W, H, FPS = 640, 360, 10
SR, CHANNELS = 48000, 1
FRAME_MS = 10
BAR_W = 40

# 四个源:颜色区分视觉,音高区分听觉(A4/C#5/E5/A5 叠起来是个 A 大三和弦)
COLORS = [(200, 60, 60), (60, 170, 90), (70, 120, 220), (215, 165, 40)]
TONES_HZ = [440.0, 554.4, 659.3, 880.0]
TONE_AMPLITUDE = 0.22  # 留足余量:四路叠加后仍不削顶


def _base_frame(index: int) -> np.ndarray:
    """底色 + 居中大数字,预生成一次;每帧只需叠加移动竖条。"""
    img = Image.new("RGBA", (W, H), (*COLORS[index % len(COLORS)], 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 200)
    except OSError:
        font = ImageFont.load_default()
    draw.text((W // 2, H // 2), str(index), font=font, fill=(255, 255, 255, 255), anchor="mm")
    return np.array(img, dtype=np.uint8)


async def _publish_video(source: rtc.VideoSource, index: int) -> None:
    base = _base_frame(index)
    interval = 1 / FPS
    i = 0
    while True:
        frame = base.copy()
        x = (i * 25) % (W - BAR_W)          # 移动竖条:证明画面是活的,不是静止图
        frame[:, x : x + BAR_W, :3] = 255
        source.capture_frame(
            rtc.VideoFrame(W, H, rtc.VideoBufferType.RGBA, frame.tobytes())
        )
        i += 1
        await asyncio.sleep(interval)


async def _publish_audio(source: rtc.AudioSource, index: int) -> None:
    """连续正弦波。两个坑都在这里:

    1. 相位必须跨帧累积,否则每帧边界会有咔哒声;
    2. **必须自己按时钟节流**。曾指望 `capture_frame` 的背压限速,
       实测源以约 2.3 倍实时速度推送,接收端每秒丢 135 帧——
       发布端要自己维持实时时钟,不能指望 SDK 替你踩刹车。
    """
    hz = TONES_HZ[index % len(TONES_HZ)]
    n = SR * FRAME_MS // 1000
    interval = FRAME_MS / 1000
    phase = 0.0
    step = 2 * np.pi * hz / SR
    next_at = asyncio.get_running_loop().time()
    while True:
        next_at += interval
        t = phase + step * np.arange(n)
        pcm = (np.sin(t) * TONE_AMPLITUDE * 32767).astype(np.int16)
        phase = (t[-1] + step) % (2 * np.pi)
        await source.capture_frame(
            rtc.AudioFrame(pcm.tobytes(), SR, CHANNELS, n)
        )
        await asyncio.sleep(max(0.0, next_at - asyncio.get_running_loop().time()))


async def main() -> None:
    index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    room_name = sys.argv[2] if len(sys.argv) > 2 else "day8"
    identity = f"src{index}"

    room = rtc.Room()
    await room.connect(URL, token(identity, room_name))
    print(f"[{identity}] 已入房 {room_name}:色 {COLORS[index % 4]}，音 {TONES_HZ[index % 4]}Hz", flush=True)

    video_source = rtc.VideoSource(W, H)
    audio_source = rtc.AudioSource(SR, CHANNELS)
    await room.local_participant.publish_track(
        rtc.LocalVideoTrack.create_video_track(f"{identity}-video", video_source),
        rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA),
    )
    await room.local_participant.publish_track(
        rtc.LocalAudioTrack.create_audio_track(f"{identity}-audio", audio_source),
        rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
    )

    try:
        await asyncio.gather(
            _publish_video(video_source, index),
            _publish_audio(audio_source, index),
        )
    finally:
        await room.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
