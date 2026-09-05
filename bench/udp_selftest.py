"""验证 netem.sh 的 pf/dummynet 规则是否真的作用于本机回环 UDP 流量。

在媒体端口范围内起一个 UDP 回显服务(源端口落在 7882-7892 → 命中下行管道),
客户端连发 N 包,统计回包丢失率与 RTT。与 WebRTC 无关,单独验证整形路径。

用法: PYTHONPATH=bench uv run python bench/udp_selftest.py [port] [count] [payload_bytes]
"""
from __future__ import annotations

import socket
import sys
import time

HOST = "10.0.0.29"
DEFAULT_PORT, DEFAULT_COUNT, DEFAULT_PAYLOAD = 7895, 200, 1200
TIMEOUT_S = 1.5


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    count = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_COUNT
    size = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_PAYLOAD

    srv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    srv.bind((HOST, port))
    srv.setblocking(False)
    cli = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cli.settimeout(0.01)

    sent_at: dict[int, float] = {}
    rtts: list[float] = []
    payload_pad = b"x" * (size - 8)

    def pump_server() -> None:
        while True:
            try:
                data, addr = srv.recvfrom(65535)
                srv.sendto(data, addr)
            except BlockingIOError:
                return

    def pump_client() -> None:
        while True:
            try:
                data, _ = cli.recvfrom(65535)
                seq = int.from_bytes(data[:8], "big")
                if seq in sent_at:
                    rtts.append(time.perf_counter() - sent_at.pop(seq))
            except socket.timeout:
                return

    for seq in range(count):
        sent_at[seq] = time.perf_counter()
        cli.sendto(seq.to_bytes(8, "big") + payload_pad, (HOST, port))
        pump_server()
        pump_client()
    deadline = time.perf_counter() + TIMEOUT_S
    while time.perf_counter() < deadline and sent_at:
        pump_server()
        pump_client()

    lost = len(sent_at)
    print(f"port={port} sent={count} recv={count - lost} loss={lost / count * 100:.1f}%")
    if rtts:
        rtts.sort()
        print(f"rtt ms: min={rtts[0]*1000:.2f} p50={rtts[len(rtts)//2]*1000:.2f} "
              f"p95={rtts[int(len(rtts)*0.95)]*1000:.2f} max={rtts[-1]*1000:.2f}")


if __name__ == "__main__":
    main()
