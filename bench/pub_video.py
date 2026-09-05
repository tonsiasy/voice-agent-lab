"""Day 5 发布端:不用摄像头,推一路合成 720p 动画并开 simulcast,每秒记录出站各层状态。

画面 = 移动色块 + 滚动噪声带。噪声是故意的:编码器压不掉它,码率才会真实地
逼近 max_bitrate,弱网实验里带宽限制才有东西可"限"。

用法: PYTHONPATH=. uv run python bench/pub_video.py [room] [csv]
"""
from __future__ import annotations

import asyncio
import csv
import sys
import time

import numpy as np
from livekit import rtc

from lkutil import URL, netem_condition, token

W, H, FPS = 1280, 720, 15
MAX_BITRATE = 2_000_000
STATS_INTERVAL = 1.0
NOISE_BAND_H = 160


def _make_frames():
    """无限生成 RGBA 帧;噪声纹理预生成一次,每帧只做切片,CPU 便宜。"""
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, size=(H * 2, W, 4), dtype=np.uint8)
    noise[..., 3] = 255
    yy, xx = np.mgrid[0:H, 0:W]
    base = np.zeros((H, W, 4), dtype=np.uint8)
    base[..., 0] = (xx * 255 // W).astype(np.uint8)
    base[..., 1] = (yy * 255 // H).astype(np.uint8)
    base[..., 3] = 255
    i = 0
    while True:
        frame = base.copy()
        off = (i * 7) % H
        frame[H - NOISE_BAND_H :, :] = noise[off : off + NOISE_BAND_H, :]
        bx = (i * 12) % (W - 200)
        frame[100:300, bx : bx + 200, :3] = (255, 80, 80)
        yield frame
        i += 1


LIMIT_NAMES = {0: "none", 1: "cpu", 2: "bandwidth", 3: "other"}


def _kind(s):
    return s.WhichOneof("stats")


class _OutboundTracker:
    """把累计计数器变成每秒增量,并按 simulcast rid 分层输出。"""

    def __init__(self, writer: csv.writer):
        self._w = writer
        self._prev: dict[str, tuple[float, int, int, int]] = {}
        self._t0 = time.monotonic()

    def feed(self, stats_list) -> None:
        now = time.monotonic()
        t = now - self._t0
        cond = netem_condition()
        remote = {s.remote_inbound_rtp.remote_inbound.local_id: s.remote_inbound_rtp
                  for s in stats_list if _kind(s) == "remote_inbound_rtp"}
        bwe = rtt = None
        for s in stats_list:
            if _kind(s) == "candidate_pair" and s.candidate_pair.candidate_pair.nominated:
                bwe = s.candidate_pair.candidate_pair.available_outgoing_bitrate
                rtt = s.candidate_pair.candidate_pair.current_round_trip_time
        parts = []
        for s in stats_list:
            if _kind(s) != "outbound_rtp" or s.outbound_rtp.stream.kind != "video":
                continue
            ob, sent, sid = s.outbound_rtp.outbound, s.outbound_rtp.sent, s.outbound_rtp.rtc.id
            rid = ob.rid or "-"
            p_t, p_bytes, p_nack, p_pli = self._prev.get(rid, (now, sent.bytes_sent, ob.nack_count, ob.pli_count))
            dt = max(now - p_t, 1e-3)
            kbps = (sent.bytes_sent - p_bytes) * 8 / dt / 1000
            d_nack, d_pli = ob.nack_count - p_nack, ob.pli_count - p_pli
            self._prev[rid] = (now, sent.bytes_sent, ob.nack_count, ob.pli_count)
            limit = LIMIT_NAMES.get(ob.quality_limitation_reason, str(ob.quality_limitation_reason))
            ri = remote.get(sid)
            lost = ri.remote_inbound.fraction_lost if ri else None
            self._w.writerow([f"{t:.1f}", cond, rid, ob.frame_width, ob.frame_height,
                              f"{ob.frames_per_second:.0f}", f"{kbps:.0f}", ob.target_bitrate,
                              limit, d_nack, d_pli,
                              f"{lost:.3f}" if lost is not None else "", bwe or "", rtt or ""])
            parts.append(f"{rid}:{ob.frame_width}x{ob.frame_height}@{ob.frames_per_second:.0f} "
                         f"{kbps:.0f}k n+{d_nack} p+{d_pli} lim={limit}")
        bwe_s = f"bwe={bwe/1000:.0f}k" if bwe else "bwe=?"
        rtt_s = f"rtt={rtt*1000:.1f}ms" if rtt else ""
        print(f"t={t:5.0f}s [{cond}] " + "  ".join(parts) + f" | {bwe_s} {rtt_s}", flush=True)


async def main() -> None:
    room_name = sys.argv[1] if len(sys.argv) > 1 else "day5"
    csv_path = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/day5-pub-{int(time.time())}.csv"

    room = rtc.Room()
    await room.connect(URL, token("publisher", room_name))
    print(f"已入房 {room_name} as publisher; csv → {csv_path}", flush=True)

    source = rtc.VideoSource(W, H)
    track = rtc.LocalVideoTrack.create_video_track("synthetic-cam", source)
    await room.local_participant.publish_track(
        track,
        rtc.TrackPublishOptions(
            source=rtc.TrackSource.SOURCE_CAMERA,
            simulcast=True,
            video_codec=rtc.VideoCodec.VP8,
            video_encoding=rtc.VideoEncoding(max_bitrate=MAX_BITRATE, max_framerate=FPS),
        ),
    )

    async def render() -> None:
        interval = 1 / FPS
        for frame in _make_frames():
            source.capture_frame(rtc.VideoFrame(W, H, rtc.VideoBufferType.RGBA, frame.tobytes()))
            await asyncio.sleep(interval)

    render_task = asyncio.create_task(render())
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "condition", "rid", "width", "height", "fps", "kbps", "target_bps",
                         "limit_reason", "d_nack", "d_pli", "fraction_lost", "bwe_bps", "rtt_s"])
        tracker = _OutboundTracker(writer)
        try:
            while True:
                await asyncio.sleep(STATS_INTERVAL)
                tracker.feed(await track.get_stats())
                f.flush()
        finally:
            render_task.cancel()
            await room.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
