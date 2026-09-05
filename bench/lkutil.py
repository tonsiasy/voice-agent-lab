"""bench 脚本共用:自签 JWT、默认地址、marker 读取。"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time

URL = os.environ.get("LIVEKIT_URL", "ws://localhost:7880")
NETEM_MARKER = "/tmp/netem.marker"


def token(identity: str, room: str, key: str = "devkey", secret: str = "secret") -> str:
    b64 = lambda d: base64.urlsafe_b64encode(d).rstrip(b"=").decode()
    now = int(time.time())
    payload = {
        "iss": key, "sub": identity, "name": identity, "nbf": now, "exp": now + 3600,
        "video": {"room": room, "roomJoin": True, "canPublish": True, "canSubscribe": True},
    }
    h = b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    p = b64(json.dumps(payload).encode())
    sig = b64(hmac.new(secret.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest())
    return f"{h}.{p}.{sig}"


def netem_condition() -> str:
    """netem.sh 每次变更写入的当前弱网条件;没有整形时为 baseline。"""
    try:
        with open(NETEM_MARKER) as f:
            return f.read().strip() or "baseline"
    except OSError:
        return "baseline"
