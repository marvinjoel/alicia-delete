"""
Servicio de Chat IA.
- Tiempo real: usa FastVLM con el frame actual de la camara.
- Historico: usa OpenAI con las descripciones guardadas en vision_ia_log.
"""
import os
import httpx
from services.database import get_vision_log
from services.vlm_service import vlm


def chat_tiempo_real(cam_id: int, pregunta: str) -> dict:
    """Responde una pregunta sobre la imagen actual de la camara via FastVLM."""
    if not vlm.modelo:
        return {"exito": 0, "mensaje": "Vision IA no esta activa. Inicia FastVLM primero."}

    img = vlm.obtener_frame_pil(cam_id)
    if img is None:
        return {"exito": 0, "mensaje": "No hay frame disponible. Inicia la camara primero."}

    # Prompt en español + pregunta del usuario
    prompt = f"{pregunta}\n\nResponde en español, breve y directo (maximo 3 oraciones)."
    respuesta = vlm.responder(img, prompt, max_tokens=180)

    if not respuesta:
        return {"exito": 0, "mensaje": "El modelo no pudo responder."}

    return {"exito": 1, "respuesta": respuesta, "modo": "tiempo_real", "modelo": vlm.nombre}


def chat_historico(cam_id: int, pregunta: str, n_eventos: int = 30) -> dict:
    """Responde preguntas sobre lo que pasó en la camara usando ChatGPT + BD."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return {"exito": 0, "mensaje": "Falta OPENAI_API_KEY en config.env"}

    # Obtener ultimas N descripciones de la camara
    logs = get_vision_log(cam_id, n_eventos)
    if not logs:
        return {"exito": 0, "mensaje": "No hay historial de Vision IA para esta camara."}

    # Armar contexto: lista de descripciones con timestamp
    contexto = "\n".join([
        f"[{log['creado_fecha']}] {log['descripcion']}" for log in reversed(logs)
    ])

    # Llamar a OpenAI
    modelo = os.getenv("OPENAI_MODELO", "gpt-4o-mini")
    try:
        resp = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": modelo,
                "messages": [
                    {"role": "system", "content": (
                        "Eres un asistente que analiza un restaurante segun el historial "
                        "de lo que la camara ha visto. Responde en español, breve y directo."
                    )},
                    {"role": "user", "content": (
                        f"Historial de la camara (ultimos {len(logs)} eventos):\n\n"
                        f"{contexto}\n\n"
                        f"Pregunta: {pregunta}"
                    )},
                ],
                "temperature": 0.3,
                "max_tokens": 400,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        respuesta = data["choices"][0]["message"]["content"].strip()
        return {"exito": 1, "respuesta": respuesta, "modo": "historico", "modelo": modelo, "eventos": len(logs)}
    except Exception as e:
        return {"exito": 0, "mensaje": f"Error OpenAI: {e}"}
