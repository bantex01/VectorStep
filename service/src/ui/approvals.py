from fastapi import APIRouter
from fastapi import Request
from fastapi.responses import HTMLResponse

from . import helpers
from .helpers import templates


router = APIRouter()


# --- lines 4660-4669 ---
@router.get("/approvals", response_class=HTMLResponse)
async def ui_approvals_list(request: Request):
    from ..executors.human import list_pending

    return templates.TemplateResponse(request, "approvals_list.html", {
        "pending": list_pending(),
        "active_page": "approvals",
    })



# --- lines 4670-4687 ---
@router.get("/approvals/{token}", response_class=HTMLResponse)
async def ui_approval(request: Request, token: str):
    from ..executors.human import get_pending_meta

    meta = get_pending_meta(token)
    if meta is None:
        return templates.TemplateResponse(request, "approval.html", {
            "state": "not_found",
            "token": token,
        })

    return templates.TemplateResponse(request, "approval.html", {
        "state": "pending",
        "token": token,
        "meta": meta,
    })



# --- lines 4688-4692 ---
@router.post("/approvals/{token}/approve", response_class=HTMLResponse)
async def ui_approval_approve(request: Request, token: str):
    return _decide(request, token, approved=True)



# --- lines 4693-4711 ---
@router.post("/approvals/{token}/reject", response_class=HTMLResponse)
async def ui_approval_reject(request: Request, token: str):
    return _decide(request, token, approved=False)


def _decide(request: Request, token: str, approved: bool):
    from ..executors.human import resolve_approval

    if not resolve_approval(token, approved):
        return templates.TemplateResponse(request, "approval.html", {
            "state": "not_found",
            "token": token,
        })

    return templates.TemplateResponse(request, "approval.html", {
        "state": "decided",
        "token": token,
        "decision": "approve" if approved else "reject",
    })

