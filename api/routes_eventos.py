from fastapi import APIRouter, Query
from services.database import get_eventos

router = APIRouter()


@router.get("/eventos")
def listar_eventos(
    limite: int = Query(default=50, ge=1, le=200),
    camara_id: int = Query(default=None),
):
    try:
        eventos = get_eventos(limite=limite, camara_id=camara_id)
        return {"ok": True, "total": len(eventos), "data": eventos}
    except Exception as e:
        return {"ok": False, "mensaje": str(e), "data": []}
