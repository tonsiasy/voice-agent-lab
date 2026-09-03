"""sherpa-onnx 流式 ASR 适配器（Day 3）。

与 Day 2 的 FasterWhisperSTT 的本质差别：
  - Day 2：`streaming=False`，框架用 VAD 切段后整段送进来 → 延迟 ∝ 内容长度
  - 本模块：`streaming=True`，逐帧喂入、边听边出增量结果 → 延迟 = 首字节时间

采样率由框架代劳：SpeechStream 声明 sample_rate=16000 后，
框架自动把房间的 48kHz 重采样成模型要的 16kHz。
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import numpy as np
import sherpa_onnx
from livekit import agents, rtc
from livekit.agents import stt, utils

SAMPLE_RATE = 16000
DEFAULT_THREADS = 2  # 实测最优：4 线程反而更慢（同步开销 > 并行收益）


def _build_recognizer(model_dir: Path, threads: int) -> sherpa_onnx.OnlineRecognizer:
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(model_dir / "tokens.txt"),
        encoder=str(model_dir / "encoder-epoch-99-avg-1.int8.onnx"),
        decoder=str(model_dir / "decoder-epoch-99-avg-1.onnx"),
        joiner=str(model_dir / "joiner-epoch-99-avg-1.int8.onnx"),
        num_threads=threads,
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        enable_endpoint_detection=True,
        # 端点规则：尾静音 2.4s（长停顿）/ 1.2s（有内容后的停顿）/ 单句上限 300s
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2,
        rule3_min_utterance_length=300,
        decoding_method="greedy_search",
    )


class SherpaStreamingSTT(stt.STT):
    """帧同步转录（RNN-T），天然流式、抗幻觉、无标点。"""

    def __init__(
        self,
        model_dir: str | os.PathLike | None = None,
        num_threads: int = DEFAULT_THREADS,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=True, interim_results=True)
        )
        d = Path(model_dir or os.environ["SHERPA_MODEL_DIR"])
        self._recognizer = _build_recognizer(d, num_threads)

    def stream(
        self,
        *,
        language: str | None = None,
        conn_options: agents.APIConnectOptions = agents.DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechStream:
        return _SherpaSpeechStream(
            stt=self, recognizer=self._recognizer, conn_options=conn_options
        )

    async def _recognize_impl(self, *args, **kwargs):  # noqa: ANN002, ANN003
        # 纯流式实现；框架在 streaming=True 时不会走这条路径
        raise NotImplementedError("SherpaStreamingSTT 只支持流式调用")


class _SherpaSpeechStream(stt.SpeechStream):
    def __init__(
        self,
        *,
        stt: SherpaStreamingSTT,
        recognizer: sherpa_onnx.OnlineRecognizer,
        conn_options: agents.APIConnectOptions,
    ) -> None:
        # sample_rate=16000 → 框架自动重采样，适配器不必自己处理格式
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=SAMPLE_RATE)
        self._recognizer = recognizer

    async def _run(self) -> None:
        rec = self._recognizer
        stream = rec.create_stream()
        last_text = ""
        speaking = False

        def _decode() -> tuple[str, bool]:
            """CPU 密集段：在线程里跑，别阻塞事件循环。"""
            while rec.is_ready(stream):
                rec.decode_stream(stream)
            return rec.get_result(stream).strip(), rec.is_endpoint(stream)

        async for data in self._input_ch:
            if not isinstance(data, rtc.AudioFrame):
                continue  # flush sentinel

            samples = (
                np.frombuffer(data.data, dtype=np.int16).astype(np.float32) / 32768.0
            )
            stream.accept_waveform(SAMPLE_RATE, samples)
            text, is_endpoint = await asyncio.to_thread(_decode)

            if text and not speaking:
                speaking = True
                self._event_ch.send_nowait(
                    stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                )

            if is_endpoint:
                if text:
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                            alternatives=[stt.SpeechData(language="zh", text=text)],
                        )
                    )
                if speaking:
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                    )
                rec.reset(stream)
                last_text, speaking = "", False
            elif text and text != last_text:
                # 增量结果：只用于显示，不喂 LLM（LLM 只吃 FINAL）
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                        alternatives=[stt.SpeechData(language="zh", text=text)],
                    )
                )
                last_text = text
