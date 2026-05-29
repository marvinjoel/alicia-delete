"""Rutas API para Vision IA (FastVLM)."""
from fastapi import APIRouter
from services.vlm_service import vlm, MODELOS

router = APIRouter(prefix="/vlm", tags=["Vision IA"])


@router.post("/{cam_id}/iniciar")
def iniciar(cam_id: int, modelo: str = "FastVLM-0.5B", intervalo: int = 10):
    """Inicia Vision IA para una camara. Carga el modelo si es necesario."""
    if modelo not in MODELOS:
        return {"exito": 0, "mensaje": f"Modelo no valido. Opciones: {list(MODELOS.keys())}"}

    if vlm.cargando:
        return {"exito": 0, "mensaje": "Modelo cargando, espere..."}

    # Si el modelo no esta cargado o es diferente, cargar primero
    if not vlm.modelo or vlm.nombre != modelo:
        vlm.cargar(modelo, cam_id=cam_id, intervalo=intervalo)
        return {"exito": 1, "mensaje": f"Descargando {modelo}...", "cargando": True}

    # Modelo ya cargado -> iniciar directamente
    vlm.iniciar_camara(cam_id, intervalo)
    return {
        "exito": 1,
        "mensaje": f"Vision IA activa ({vlm.nombre})",
        "modelo": vlm.nombre,
        "device": vlm.device,
    }


@router.post("/{cam_id}/detener")
def detener(cam_id: int):
    """Detiene Vision IA para una camara."""
    vlm.detener_camara(cam_id)
    return {"exito": 1, "mensaje": "Vision IA detenida"}


@router.get("/estado")
def estado():
    """Estado actual del servicio VLM."""
    return {
        "modelo": vlm.nombre,
        "cargando": vlm.cargando,
        "device": vlm.device,
        "modelos_disponibles": list(MODELOS.keys()),
        "camaras_activas": [k for k, v in vlm._cams.items() if v.get("activo")],
    }


@router.get("/{cam_id}/historial")
def historial(cam_id: int, limite: int = 30):
    """Ultimas N descripciones de Vision IA para una camara."""
    from services.database import get_vision_log
    return get_vision_log(cam_id, limite)


@router.post("/{cam_id}/chat")
def chat(cam_id: int, pregunta: str, modo: str = "tiempo_real"):
    """Chat sobre la camara.
    modo=tiempo_real: FastVLM analiza el frame actual.
    modo=historico:   ChatGPT responde basado en vision_ia_log.
    """
    from services.chat_service import chat_tiempo_real, chat_historico
    if modo == "historico":
        return chat_historico(cam_id, pregunta)
    return chat_tiempo_real(cam_id, pregunta)
