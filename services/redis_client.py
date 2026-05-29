import os
import redis

_cliente = None


def get_redis() -> redis.Redis:
    global _cliente
    if _cliente is None:
        _cliente = redis.Redis(
            host=os.getenv("REDIS_HOST", "127.0.0.1"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            decode_responses=True,
        )
    return _cliente


def puede_alertar(camara_id: int, tipo_evento: str) -> bool:
    """Retorna True si no se envio alerta reciente para este evento/camara."""
    cooldown = int(os.getenv("ALERTA_COOLDOWN_MIN", 5)) * 60
    clave = f"alerta:{camara_id}:{tipo_evento}"
    r = get_redis()
    if r.exists(clave):
        return False
    r.setex(clave, cooldown, "1")
    return True


def set_camara_activa(camara_id: int):
    get_redis().set(f"camara:activa:{camara_id}", "1")


def set_camara_inactiva(camara_id: int):
    get_redis().delete(f"camara:activa:{camara_id}")


def is_camara_activa(camara_id: int) -> bool:
    return get_redis().exists(f"camara:activa:{camara_id}") == 1
