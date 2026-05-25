"""Lightweight one-shot helper for calling the OpenClaw Gateway WebSocket API."""
import asyncio
import base64
import json
import logging
import time
import uuid
from pathlib import Path

import websockets
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
    load_pem_private_key,
)

logger = logging.getLogger(__name__)

_IDENTITY_PATH = Path.home() / ".openclaw" / "identity" / "device.json"
_DEVICE_AUTH_PATH = Path.home() / ".openclaw" / "identity" / "device-auth.json"
GATEWAY_WS_URL = "ws://127.0.0.1:18789/rpc"


def _load_identity() -> tuple[str, object, str]:
    data = json.loads(_IDENTITY_PATH.read_text())
    priv_key = load_pem_private_key(data["privateKeyPem"].encode(), password=None)
    pub_raw = priv_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    pub_b64url = base64.urlsafe_b64encode(pub_raw).rstrip(b"=").decode()
    return data["deviceId"], priv_key, pub_b64url


def _load_operator_token() -> tuple[str, list[str]]:
    data = json.loads(_DEVICE_AUTH_PATH.read_text())
    entry = data["tokens"]["operator"]
    return entry["token"], entry["scopes"]


def _sign_payload(
    device_id: str, priv_key, op_token: str, scopes: list[str], nonce: str
) -> tuple[str, int]:
    signed_at = int(time.time() * 1000)
    payload_str = "|".join([
        "v3", device_id, "gateway-client", "backend", "operator",
        ",".join(scopes), str(signed_at), op_token, nonce, "darwin", "",
    ])
    sig = base64.b64encode(priv_key.sign(payload_str.encode())).decode()
    return sig, signed_at


async def gateway_call(method: str, params: dict, gateway_url: str = GATEWAY_WS_URL) -> dict:
    """Open a gateway WS connection, authenticate, invoke one method, return payload."""
    device_id, priv_key, pub_b64url = _load_identity()
    op_token, scopes = _load_operator_token()

    async with websockets.connect(gateway_url) as ws:
        challenge = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        nonce = challenge["payload"]["nonce"]
        sig, signed_at = _sign_payload(device_id, priv_key, op_token, scopes, nonce)

        connect_id = str(uuid.uuid4())
        await ws.send(json.dumps({
            "type": "req", "id": connect_id, "method": "connect",
            "params": {
                "minProtocol": 3, "maxProtocol": 3,
                "client": {
                    "id": "gateway-client", "version": "2026.4.11",
                    "platform": "darwin", "mode": "backend",
                },
                "role": "operator",
                "scopes": scopes,
                "device": {
                    "id": device_id, "publicKey": pub_b64url,
                    "signature": sig, "signedAt": signed_at, "nonce": nonce,
                },
                "auth": {"token": op_token},
            },
        }))
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=30)
            msg = json.loads(raw)
            if msg.get("type") == "res" and msg.get("id") == connect_id:
                if not msg.get("ok"):
                    raise RuntimeError(
                        f"Gateway connect failed: {msg.get('error', {}).get('message', msg)}"
                    )
                break

        req_id = str(uuid.uuid4())
        await ws.send(json.dumps({
            "type": "req", "id": req_id, "method": method, "params": params,
        }))
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=30)
            msg = json.loads(raw)
            if msg.get("type") == "res" and msg.get("id") == req_id:
                if not msg.get("ok"):
                    raise RuntimeError(
                        f"Gateway {method} failed: {msg.get('error', {}).get('message', msg)}"
                    )
                return msg.get("payload") or {}


async def gateway_call_safe(
    method: str, params: dict, gateway_url: str = GATEWAY_WS_URL
) -> dict | None:
    """Like gateway_call but returns None on any error instead of raising."""
    try:
        return await gateway_call(method, params, gateway_url)
    except Exception as exc:
        logger.debug("gateway_call_safe(%s) failed: %s", method, exc)
        return None
