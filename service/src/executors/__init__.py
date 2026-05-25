from .human import HumanExecutor
from .openclaw_ws import OpenClawWSExecutor
from .webhook import WebhookExecutor

EXECUTORS: dict = {
    "openclaw": OpenClawWSExecutor,
    "webhook": WebhookExecutor,
    "human": HumanExecutor,
}
