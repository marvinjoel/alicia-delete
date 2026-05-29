import os
import httpx


def enviar_alerta(mensaje: str, numero: str = None) -> bool:
    api_key = os.getenv("WASENGER_API_KEY", "")
    destino = numero or os.getenv("WASENGER_NUMERO_DESTINO", "")

    if not api_key:
        print("[Wasenger] API Key no configurada — mensaje no enviado.")
        return False

    if not destino:
        print("[Wasenger] Numero destino no configurado — mensaje no enviado.")
        return False

    try:
        response = httpx.post(
            "https://api.wassenger.com/v1/messages",
            headers={
                "token": api_key,
                "Content-Type": "application/json",
            },
            json={"phone": destino, "message": mensaje},
            timeout=10,
        )
        ok = response.status_code in (200, 201)
        if not ok:
            print(f"[Wasenger] Error HTTP {response.status_code}: {response.text}")
        return ok
    except Exception as e:
        print(f"[Wasenger] Excepcion al enviar: {e}")
        return False
