import threading

from fastapi import APIRouter
from workers.camera_worker import workers_activos, CameraWorker

router = APIRouter()


@router.post("/camaras/{camara_id}/iniciar")
def iniciar_camara(camara_id: int):
    if camara_id in workers_activos and workers_activos[camara_id].corriendo:
        return {"ok": False, "mensaje": "La camara ya esta procesando"}

    # Lanzar en thread separado para no bloquear uvicorn mientras carga YOLO
    def _arrancar():
        try:
            worker = CameraWorker(camara_id)
            worker.iniciar()
            workers_activos[camara_id] = worker
        except Exception as e:
            print(f"[iniciar {camara_id}] Error: {e}")

    threading.Thread(target=_arrancar, daemon=True, name=f"init-{camara_id}").start()
    return {"ok": True, "mensaje": f"Iniciando IA para camara {camara_id}..."}


@router.post("/camaras/{camara_id}/detener")
def detener_camara(camara_id: int):
    worker = workers_activos.get(camara_id)
    if not worker:
        return {"ok": False, "mensaje": "La camara no esta activa"}
    worker.detener()
    del workers_activos[camara_id]
    return {"ok": True, "mensaje": f"IA detenida para camara {camara_id}"}


@router.get("/camaras")
def listar_camaras():
    return {
        "activas": [
            {"camara_id": cid, "corriendo": w.corriendo}
            for cid, w in workers_activos.items()
        ]
    }
