"""语音 Agent:LiveKit Agents 闭环（Day 3 起改用流式 ASR）。

管道:silero VAD(本地) → sherpa-onnx 流式 ASR(本地,帧同步 RNN-T)
     → DeepSeek LLM(openai 兼容) → edge-tts TTS(免 key,自定义适配器)

STT 可用 STT_BACKEND=whisper 切回 Day 2 的非流式实现做对照。

运行:
    uv run python agent.py download-files   # 预下载 VAD/turn-detector 模型
    uv run python agent.py connect --room day1
"""
from __future__ import annotations

import asyncio
import io
import os

from dotenv import load_dotenv

load_dotenv()

import edge_tts
from faster_whisper import WhisperModel
from livekit import agents, rtc
from livekit.agents import Agent, AgentSession, stt, tts, utils
from livekit.plugins import openai, silero

from sherpa_stt import SherpaStreamingSTT

STT_BACKEND = os.environ.get("STT_BACKEND", "sherpa")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
EDGE_VOICE = os.environ.get("EDGE_VOICE", "zh-CN-XiaoxiaoNeural")
TTS_SAMPLE_RATE = 24000


class FasterWhisperSTT(stt.STT):
    """非流式 STT:AgentSession 会用 VAD 自动切段后逐段调用本类。"""

    def __init__(self, model_size: str = WHISPER_MODEL):
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )
        # int8 量化走 CPU;首次调用会从 HF(或 HF_ENDPOINT 镜像)拉模型
        self._model = WhisperModel(model_size, device="cpu", compute_type="int8")

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: str | None,
        conn_options: agents.APIConnectOptions,
    ) -> stt.SpeechEvent:
        frame = rtc.combine_audio_frames(buffer)
        wav = io.BytesIO(frame.to_wav_bytes())

        def _transcribe() -> str:
            segments, _info = self._model.transcribe(
                wav, language=language or "zh", beam_size=1, vad_filter=False
            )
            return "".join(s.text for s in segments).strip()

        text = await asyncio.to_thread(_transcribe)
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(language=language or "zh", text=text)],
        )


class EdgeTTS(tts.TTS):
    """edge-tts 输出 mp3 分片;推给 AudioEmitter 由框架解码(依赖 av)。"""

    def __init__(self, voice: str = EDGE_VOICE):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=TTS_SAMPLE_RATE,
            num_channels=1,
        )
        self._voice = voice

    def synthesize(
        self,
        text: str,
        *,
        conn_options: agents.APIConnectOptions = agents.DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        return _EdgeChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class _EdgeChunkedStream(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=TTS_SAMPLE_RATE,
            num_channels=1,
            mime_type="audio/mp3",
        )
        communicate = edge_tts.Communicate(self.input_text, self._tts._voice)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                output_emitter.push(chunk["data"])
        output_emitter.flush()


def _build_stt() -> agents.stt.STT:
    """默认流式(Day 3);STT_BACKEND=whisper 可切回非流式做 A/B 对照。"""
    if STT_BACKEND == "whisper":
        print("[stt] 非流式 faster-whisper（对照组）", flush=True)
        return FasterWhisperSTT()
    print("[stt] 流式 sherpa-onnx zipformer", flush=True)
    return SherpaStreamingSTT()


async def entrypoint(ctx: agents.JobContext) -> None:
    await ctx.connect()
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=_build_stt(),
        llm=openai.LLM(
            model=os.environ.get("LLM_MODEL", "deepseek-chat"),
            base_url="https://api.deepseek.com/v1",
            api_key=os.environ["DEEPSEEK_API_KEY"],
        ),
        tts=EdgeTTS(),
    )
    print("[probe] session.start() begin", flush=True)
    await session.start(
        room=ctx.room,
        agent=Agent(
            instructions=(
                "你是语音助手「小融」。用口语化的中文回答,每次不超过两句话,"
                "不用任何列表、代码或表情符号——你的输出会被直接转成语音。"
            )
        ),
    )
    print("[probe] session.start() done, greeting...", flush=True)
    await session.generate_reply(instructions="用一句话向用户问好并自我介绍")
    print("[probe] greeting done", flush=True)


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
