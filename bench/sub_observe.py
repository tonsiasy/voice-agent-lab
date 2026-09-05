"""Day 5 观测端:订阅房间里的视频,每秒采样接收侧 QoS 指标并落 CSV。

看的就是 SFU 面对弱网时"做了什么":
  d_nack / d_pli   —— 观测端向 SFU 要重传 / 要关键帧的次数(每秒增量)
  lost             —— 累计丢包(SFU→观测端这一跳,已扣除 NACK 补回的)
  width x height   —— 当前拿到的 simulcast 层;下行带宽不够时 SFU 会给低层
  freeze           —— 画面冻结次数(用户可感知的最终后果)
  rtt              —— ICE 候选对上的往返(能直接看到 delay 整形是否生效)

用法: PYTHONPATH=. uv run python bench/sub_observe.py [room] [csv]
"""
from __future__ import annotations

import asyncio
import csv
import sys
import time

from livekit import rtc

from lkutil import URL, netem_condition, token

STATS_INTERVAL = 1.0


def _kind(s):
    return s.WhichOneof("stats")


class _InboundTracker:
    def __init__(self, writer: csv.writer):
        self._w = writer
        self._prev: tuple[float, int, int, int, int, int] | None = None
        self._t0 = time.monotonic()
        self.decoded_frames = 0

    def feed(self, stats_list) -> None:
        now = time.monotonic()
        t = now - self._t0
        cond = netem_condition()
        rtt = None
        for s in stats_list:
            if _kind(s) == "candidate_pair" and s.candidate_pair.candidate_pair.nominated:
                rtt = s.candidate_pair.candidate_pair.current_round_trip_time
        for s in stats_list:
            if _kind(s) != "inbound_rtp" or s.inbound_rtp.stream.kind != "video":
                continue
            ib, rcv = s.inbound_rtp.inbound, s.inbound_rtp.received
            cur = (now, ib.bytes_received, ib.nack_count, ib.pli_count, rcv.packets_lost, ib.frames_decoded)
            p = self._prev or cur
            dt = max(now - p[0], 1e-3)
            kbps = (cur[1] - p[1]) * 8 / dt / 1000
            d_nack, d_pli, d_lost, d_frames = cur[2] - p[2], cur[3] - p[3], cur[4] - p[4], cur[5] - p[5]
            self._prev = cur
            self._w.writerow([f"{t:.1f}", cond, ib.frame_width, ib.frame_height, f"{d_frames/dt:.0f}",
                              f"{kbps:.0f}", d_lost, rcv.packets_lost, d_nack, d_pli,
                              ib.freeze_count, f"{ib.total_freeze_duration:.2f}",
                              f"{rcv.jitter*1000:.1f}", f"{rtt*1000:.1f}" if rtt else ""])
            rtt_s = f"rtt={rtt*1000:.1f}ms" if rtt else ""
            print(f"t={t:5.0f}s [{cond}] {ib.frame_width}x{ib.frame_height}@{d_frames/dt:.0f} "
                  f"{kbps:.0f}k lost+{d_lost}(Σ{rcv.packets_lost}) nack+{d_nack} pli+{d_pli} "
                  f"freeze={ib.freeze_count}/{ib.total_freeze_duration:.1f}s "
                  f"jit={rcv.jitter*1000:.1f}ms {rtt_s}", flush=True)


class _TrackHolder:
    """弱网实验会打断连接,重连后轨对象是新的——必须跟着换,否则统计永远停在旧轨。"""

    def __init__(self) -> None:
        self.track: rtc.RemoteVideoTrack | None = None
        self._drain: asyncio.Task | None = None

    def bind(self, track: rtc.RemoteVideoTrack) -> None:
        if self._drain:
            self._drain.cancel()
        self.track = track

        async def drain() -> None:
            # 必须真的消费帧,解码器才会跑,frames_decoded / freeze 才有意义
            try:
                async for _ in rtc.VideoStream(track):
                    pass
            except Exception:
                pass

        self._drain = asyncio.create_task(drain())

    def unbind(self, track) -> None:
        if self.track is track:
            self.track = None

    def close(self) -> None:
        if self._drain:
            self._drain.cancel()


async def main() -> None:
    room_name = sys.argv[1] if len(sys.argv) > 1 else "day5"
    csv_path = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/day5-sub-{int(time.time())}.csv"

    room = rtc.Room()
    holder = _TrackHolder()

    @room.on("track_subscribed")
    def _on_track(track, pub, participant):
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            print(f"[bind] 订阅到 {participant.identity} 的视频轨 {pub.sid}", flush=True)
            holder.bind(track)

    @room.on("track_unsubscribed")
    def _off_track(track, pub, participant):
        print(f"[unbind] 失去视频轨 {pub.sid}", flush=True)
        holder.unbind(track)

    await room.connect(URL, token("observer", room_name))
    print(f"已入房 {room_name} as observer; csv → {csv_path}", flush=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "condition", "width", "height", "fps", "kbps", "d_lost", "lost_total",
                         "d_nack", "d_pli", "freeze_count", "freeze_s", "jitter_ms", "rtt_ms"])
        tracker = _InboundTracker(writer)
        try:
            while True:
                await asyncio.sleep(STATS_INTERVAL)
                track = holder.track
                if track is None:
                    continue
                try:
                    tracker.feed(await track.get_stats())
                except Exception as exc:  # 轨在重连中失效,等下一次 track_subscribed
                    print(f"[warn] get_stats 失败({exc.__class__.__name__}),等待重新订阅", flush=True)
                    continue
                f.flush()
        finally:
            holder.close()
            await room.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
