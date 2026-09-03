"""端到端探针：以参与者身份入房，把 wav 当麦克风推进去，量真实链路延迟。

不依赖真人麦克风，用于回归验证与延迟对比（Day 2 whisper vs Day 3 流式）。
用法: PYTHONPATH=. uv run python bench/e2e_probe.py [wav] [静音秒数]
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import sys
import time
import wave

import numpy as np
from dotenv import load_dotenv
from livekit import rtc

load_dotenv()

URL = "ws://localhost:7880"
ROOM = "day1"
SR = 48000  # 房间音频采样率；框架会替 STT 重采样到 16k
FRAME_MS = 10


def token(identity: str, key: str = "devkey", secret: str = "secret") -> str:
    b64 = lambda d: base64.urlsafe_b64encode(d).rstrip(b"=").decode()
    now = int(time.time())
    payload = {
        "iss": key, "sub": identity, "name": identity, "nbf": now, "exp": now + 3600,
        "video": {"room": ROOM, "roomJoin": True, "canPublish": True, "canSubscribe": True},
    }
    h = b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    p = b64(json.dumps(payload).encode())
    sig = b64(hmac.new(secret.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest())
    return f"{h}.{p}.{sig}"


def load_and_resample(path: str) -> np.ndarray:
    """读 16k wav → 线性插值到 48k（探针够用，非生产重采样）。"""
    with wave.open(path) as w:
        src_sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if src_sr == SR:
        return pcm
    idx = np.linspace(0, len(pcm) - 1, int(len(pcm) * SR / src_sr))
    return np.interp(idx, np.arange(len(pcm)), pcm).astype(np.int16)


async def main() -> None:
    wav = sys.argv[1] if len(sys.argv) > 1 else "bench/test-zh.wav"
    tail_s = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0

    speech = load_and_resample(wav)
    samples = np.concatenate([speech, np.zeros(int(SR * tail_s), dtype=np.int16)])

    room = rtc.Room()
    agent_audio_at: list[float] = []

    @room.on("track_subscribed")
    def _on_track(track, pub, participant):
        if track.kind == rtc.TrackKind.KIND_AUDIO and "agent" in participant.identity:
            async def watch():
                stream = rtc.AudioStream(track)
                last_voice = 0.0
                async for ev in stream:
                    buf = np.frombuffer(ev.frame.data, dtype=np.int16)
                    now = time.perf_counter()
                    if np.abs(buf).max() > 500:
                        # 距上次发声 >1s 视为一次新的开口(去抖,区分开场白与回答)
                        if now - last_voice > 1.0:
                            agent_audio_at.append(now)
                        last_voice = now
            asyncio.create_task(watch())

    await room.connect(URL, token("probe"))
    print(f"已入房 {ROOM}，在场 {len(room.remote_participants) + 1} 人")

    source = rtc.AudioSource(SR, 1)
    track = rtc.LocalAudioTrack.create_audio_track("probe-mic", source)
    await room.local_participant.publish_track(
        track,
        rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
    )
    await asyncio.sleep(2.0)   # 等 Agent 订阅

    step = SR * FRAME_MS // 1000
    t_start = time.perf_counter()
    speech_end: float | None = None
    for i in range(0, len(samples), step):
        chunk = samples[i : i + step]
        if len(chunk) < step:
            break
        if speech_end is None and i >= len(speech):
            speech_end = time.perf_counter()
            print(f"[t={speech_end - t_start:6.2f}s] 语音播完，开始送静音")
        await source.capture_frame(
            rtc.AudioFrame(chunk.tobytes(), SR, 1, len(chunk))
        )
        await asyncio.sleep(FRAME_MS / 1000)

    # 等「语音播完之后」的开口——不能用 agent_audio_at 非空判断,
    # 开场白早已把它填上了
    deadline = time.perf_counter() + 25
    while time.perf_counter() < deadline:
        if speech_end and any(t > speech_end for t in agent_audio_at):
            break
        await asyncio.sleep(0.1)

    print("\n=== 端到端延迟 ===")
    after = [t for t in agent_audio_at if speech_end and t > speech_end]
    print(f"Agent 开口次数: {len(agent_audio_at)}（含入场开场白）")
    if after and speech_end:
        print(f"★ 说完 → Agent 出声: {after[0] - speech_end:.2f} s")
    else:
        print("未捕获到「说完之后」的 Agent 语音（超时 25s）")
    await room.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
