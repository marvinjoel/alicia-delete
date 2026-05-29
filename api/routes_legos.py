import os
import json

from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.post("/legos/{lego_id}/embedding")
def generar_embedding(lego_id: int):
    """
    Genera y guarda el embedding facial de un Lego a partir de su foto de perfil.
    Requiere deepface instalado: pip install deepface
    """
    from services.database import get_connection, actualizar_embedding

    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT rowid, nombre, foto FROM legos WHERE rowid = %s AND activo = 1",
            (lego_id,),
        )
        lego = cur.fetchone()
    conn.close()

    if not lego:
        raise HTTPException(status_code=404, detail="Lego no encontrado")
    if not lego.get("foto"):
        raise HTTPException(status_code=400, detail="El Lego no tiene foto de perfil")

    base = os.getenv("PHP_UPLOADS_PATH", r"d:\Instalaciones\laragon\www\alicia")
    foto_path = os.path.join(base, lego["foto"].replace("/", os.sep))

    if not os.path.exists(foto_path):
        raise HTTPException(
            status_code=400,
            detail=f"Foto no encontrada: {foto_path}. Revisa PHP_UPLOADS_PATH en config.env",
        )

    try:
        from deepface import DeepFace

        result = DeepFace.represent(
            img_path=foto_path,
            model_name="Facenet",
            enforce_detection=True,
        )
        embedding = result[0]["embedding"]
        actualizar_embedding(lego_id, embedding)
        print(f"[routes_legos] Embedding generado para Lego {lego_id} ({lego['nombre']}) — {len(embedding)} dims")
        return {
            "ok": True,
            "mensaje": f"Embedding generado para {lego['nombre']}",
            "dims": len(embedding),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando embedding: {str(e)}")


@router.delete("/legos/{lego_id}/embedding")
def borrar_embedding(lego_id: int):
    """Elimina el embedding facial guardado (para regenerar o cambiar a vestimenta)."""
    from services.database import actualizar_embedding
    actualizar_embedding(lego_id, None)
    return {"ok": True, "mensaje": "Embedding eliminado"}


@router.get("/legos/activos")
def listar_legos_activos():
    """Lista todos los Legos activos con estado del embedding (sin devolver el vector)."""
    from services.database import get_legos_activos
    legos = get_legos_activos()
    for lego in legos:
        lego["tiene_embedding"] = lego.get("embedding_facial") is not None
        lego.pop("embedding_facial", None)
    return legos
