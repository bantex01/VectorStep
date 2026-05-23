from .human import HumanExecutor
from .openclaw import OpenClawExecutor
from .webhook import WebhookExecutor

EXECUTORS: dict = {
    "openclaw": OpenClawExecutor,
    "webhook": WebhookExecutor,
    "human": HumanExecutor,
}
