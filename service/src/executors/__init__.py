from .gateway import GatewayExecutor
from .human import HumanExecutor
from .openclaw_ws import OpenClawWSExecutor
from .webhook import WebhookExecutor

EXECUTORS: dict = {
    "openclaw": OpenClawWSExecutor,
    "gateway": GatewayExecutor,
    "webhook": WebhookExecutor,
    "human": HumanExecutor,
}