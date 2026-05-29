import time
from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse, Response
from workers.camera_worker import workers_activos

router = APIRouter()


def _generar_mjpeg(camara_id: int):
    """
    Generador infinito de frames JPEG.
    El browser recibe el stream con Content-Type: multipart/x-mixed-replace
    y lo muestra en un <img> tag como si fuera video.
    """
    while True:
        worker = workers_activos.get(camara_id)

        if worker and worker.frame_actual:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + worker.frame_actual
                + b"\r\n"
            )
        else:
            # Camara no activa: enviar frame negro de espera
            import cv2
            import numpy as np
            frame_espera = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(
                frame_espera,
                f"Camara {camara_id} — IA no iniciada",
                (60, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (100, 100, 100),
                2,
            )
            _, jpeg = cv2.imencode(".jpg", frame_espera)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg.tobytes()
                + b"\r\n"
            )

        time.sleep(0.04)  # ~25 fps maximo


@router.get("/camaras/{camara_id}/stream")
def stream_camara(camara_id: int):
    return StreamingResponse(
        _generar_mjpeg(camara_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/camaras/{camara_id}/frame")
def frame_camara(camara_id: int):
    """Devuelve el ultimo frame JPEG como imagen estatica. Usado por el VLM del navegador."""
    worker = workers_activos.get(camara_id)
    if worker and worker.frame_actual:
        return Response(
            content=worker.frame_actual,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )
    return Response(status_code=404)
