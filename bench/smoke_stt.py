"""冒烟测试:不起 LiveKit,直接把 wav 按 20ms 帧喂进流式适配器,验证事件序列。"""
import asyncio, wave, numpy as np
from dotenv import load_dotenv; load_dotenv()
from livekit import rtc
from livekit.agents import stt as sttmod
from sherpa_stt import SherpaStreamingSTT

async def main():
    engine = SherpaStreamingSTT()
    stream = engine.stream()
    with wave.open("bench/test-zh.wav") as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    step = 320  # 20ms @16k

    # 语音后补 3s 静音:端点检测靠「持续收到静音帧」判定说完,
    # 真实管道里麦克风会一直送帧,测试须还原这个条件
    silence = np.zeros(16000 * 3, dtype=np.int16)
    full = np.concatenate([pcm, silence])

    async def feed():
        for i in range(0, len(full), step):
            chunk = full[i:i+step]
            stream.push_frame(rtc.AudioFrame(
                data=chunk.tobytes(), sample_rate=16000,
                num_channels=1, samples_per_channel=len(chunk)))
            await asyncio.sleep(0)      # 让出事件循环
        await asyncio.sleep(1.0)
        stream.end_input()

    counts, finals = {}, []
    async def drain():
        async for ev in stream:
            counts[ev.type.value] = counts.get(ev.type.value, 0) + 1
            if ev.type == sttmod.SpeechEventType.FINAL_TRANSCRIPT:
                finals.append(ev.alternatives[0].text)

    await asyncio.wait_for(asyncio.gather(feed(), drain()), timeout=90)
    print("事件统计:", counts)
    print("FINAL 条数:", len(finals))
    for t in finals: print("  →", t)

asyncio.run(main())
