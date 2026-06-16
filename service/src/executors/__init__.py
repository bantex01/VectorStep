from .gateway import GatewayExecutor
from .human import HumanExecutor
from .openclaw_ws import OpenClawWSExecutor
from .pipeline import PipelineExecutor
from .webhook import WebhookExecutor

EXECUTORS: dict = {
    "openclaw": OpenClawWSExecutor,
    "gateway": GatewayExecutor,
    "webhook": WebhookExecutor,
    "human": HumanExecutor,
    "pipeline": PipelineExecutor,
}