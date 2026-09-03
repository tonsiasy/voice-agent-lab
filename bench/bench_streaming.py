"""流式 ASR 基准：模拟真实逐帧喂入，测 RTF 与增量延迟。

用法: uv run python bench/bench_streaming.py <模型目录> <wav文件> [线程数]
"""
from __future__ import annotations

import resource
import sys
import time
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx

CHUNK_MS = 100  # 每次喂 100ms,模拟实时到帧


def load_wav16k(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as w:
        assert w.getnchannels() == 1, "需要单声道"
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0, sr


def build(model_dir: Path, threads: int) -> sherpa_onnx.OnlineRecognizer:
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(model_dir / "tokens.txt"),
        encoder=str(model_dir / "encoder-epoch-99-avg-1.int8.onnx"),
        decoder=str(model_dir / "decoder-epoch-99-avg-1.onnx"),
        joiner=str(model_dir / "joiner-epoch-99-avg-1.int8.onnx"),
        num_threads=threads,
        sample_rate=16000,
        feature_dim=80,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2,
        rule3_min_utterance_length=300,
        decoding_method="greedy_search",
    )


def main() -> None:
    model_dir = Path(sys.argv[1])
    wav_path = sys.argv[2]
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 2

    t0 = time.perf_counter()
    rec = build(model_dir, threads)
    load_s = time.perf_counter() - t0

    samples, sr = load_wav16k(wav_path)
    dur_s = len(samples) / sr
    chunk_n = int(sr * CHUNK_MS / 1000)

    stream = rec.create_stream()
    chunk_latencies: list[float] = []
    first_text_at: float | None = None
    partials: list[str] = []

    proc_start = time.perf_counter()
    for i in range(0, len(samples), chunk_n):
        c0 = time.perf_counter()
        stream.accept_waveform(sr, samples[i : i + chunk_n])
        while rec.is_ready(stream):
            rec.decode_stream(stream)
        chunk_latencies.append((time.perf_counter() - c0) * 1000)
        txt = rec.get_result(stream)
        if txt and first_text_at is None:
            first_text_at = (i + chunk_n) / sr  # 音频进行到第几秒出了首字
        if txt and (not partials or partials[-1] != txt):
            partials.append(txt)

    stream.input_finished()
    while rec.is_ready(stream):
        rec.decode_stream(stream)
    final = rec.get_result(stream)
    proc_s = time.perf_counter() - proc_start

    lat = sorted(chunk_latencies)
    p = lambda q: lat[min(int(len(lat) * q), len(lat) - 1)]
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2

    print(f"模型加载      : {load_s:.2f}s   线程 {threads}")
    print(f"音频时长      : {dur_s:.2f}s")
    print(f"总处理耗时    : {proc_s:.2f}s")
    print(f"★ RTF        : {proc_s / dur_s:.3f}   (<1 才可实时, <0.3 为宜)")
    print(f"每 100ms 块延迟: P50 {p(0.5):.1f}ms  P95 {p(0.95):.1f}ms  max {lat[-1]:.1f}ms")
    print(f"              (需 < {CHUNK_MS}ms 才追得上实时)")
    if first_text_at:
        print(f"首字出现于音频 : {first_text_at:.2f}s")
    print(f"进程峰值内存  : {peak_mb:.0f} MB")
    print(f"中间结果次数  : {len(partials)}")
    print(f"最终转写      : {final}")


if __name__ == "__main__":
    main()
