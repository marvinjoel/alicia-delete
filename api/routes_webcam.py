import cv2
import numpy as np
import os

from fastapi import APIRouter, Request, Query
from fastapi.responses import Response
from ultralytics import YOLO

router = APIRouter()

_modelo = None


def _get_modelo():
    global _modelo
    if _modelo is None:
        nombre = os.getenv("MODELO_YOLO_DEFAULT", "yolo11m.pt")
        print(f"[Webcam] Cargando modelo: {nombre}")
        _modelo = YOLO(f"models/{nombre}")
    return _modelo


@router.post("/webcam/frame")
async def analizar_frame_webcam(
    request: Request,
    camara_id: int = Query(0, description="ID de la camara en BD"),
):
    """
    Recibe un frame JPEG enviado desde el navegador del cliente.
    Si existe un worker activo para camara_id, usa su modelo YOLO y pasa el
    frame por todos los procesadores de analitica configurados en BD
    (igual que el modo YouTube/RTSP), generando eventos y alertas.
    Actualiza worker.frame_actual para que el stream MJPEG lo sirva.
    Si no hay worker, hace solo inferencia YOLO sin analitica (fallback).
    """
    from workers.camera_worker import workers_activos

    data = await request.body()
    arr  = np.frombuffer(data, np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        return Response(status_code=400)

    worker = workers_activos.get(camara_id)

    # Siempre guardar el frame limpio en el worker para que FastVLM lo use,
    # aunque no haya procesadores YOLO configurados.
    if worker:
        worker.frame_raw = img

    if not worker:
        print(f"[Webcam] Cam={camara_id} | AVISO: worker no encontrado — usando fallback sin analitica")
    elif not worker.procesadores:
        print(f"[Webcam] Cam={camara_id} | AVISO: worker sin procesadores — usando fallback sin analitica")

    if worker and worker.procesadores:
        # Usar el mismo modelo y procesadores que el worker (igual que modo YouTube)
        res       = worker.modelo(img, verbose=False)
        frame_out = res[0].plot()

        for proc in worker.procesadores:
            try:
                frame_out = proc.procesar(frame_out, res)
            except Exception as e:
                print(f"[Webcam] Cam={camara_id} | ERROR en {proc.__class__.__name__}: {e}")

        _, jpeg = cv2.imencode(".jpg", frame_out, [cv2.IMWRITE_JPEG_QUALITY, 72])
        jpeg_bytes = jpeg.tobytes()

        # Actualizar frame_actual para que /camaras/{id}/stream lo sirva
        worker.frame_actual = jpeg_bytes

        return Response(content=jpeg_bytes, media_type="image/jpeg")

    # Fallback: solo YOLO sin procesadores de analitica
    res = _get_modelo()(img, verbose=False)
    out = res[0].plot()

    _, jpeg = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 72])
    return Response(content=jpeg.tobytes(), media_type="image/jpeg")
