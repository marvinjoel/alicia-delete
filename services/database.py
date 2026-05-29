import os
import json
import pymysql
import pymysql.cursors


def get_connection():
    return pymysql.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", 3306)),
        db=os.getenv("DB_NAME", "alicia_ia"),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASS", ""),
        charset="utf8mb4",
        collation="utf8mb4_general_ci",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )


def get_camara(camara_id: int) -> dict | None:
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM camaras WHERE rowid = %s LIMIT 1", (camara_id,))
        return cur.fetchone()
    conn.close()


def get_analiticas_camara(camara_id: int) -> list:
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            """SELECT ca.*, dm.params_schema AS params_schema
               FROM camara_analiticas ca
               LEFT JOIN diccionario_modulos dm
                 ON dm.clave COLLATE utf8mb4_general_ci = ca.modulo COLLATE utf8mb4_general_ci
               WHERE ca.fk_camara = %s AND ca.activo = 1""",
            (camara_id,),
        )
        rows = cur.fetchall()
    conn.close()
    # Parsear config_json/params_schema y usar params_schema como default
    for row in rows:
        if row.get("config_json") and isinstance(row["config_json"], str):
            try:
                row["config_json"] = json.loads(row["config_json"])
            except Exception:
                row["config_json"] = {}
        if row.get("params_schema") and isinstance(row["params_schema"], str):
            try:
                row["params_schema"] = json.loads(row["params_schema"])
            except Exception:
                row["params_schema"] = {}
        if not row.get("config_json") and row.get("params_schema"):
            # Extraer solo los valores "default" del schema para obtener config plana
            defaults = {}
            for key, definition in row["params_schema"].items():
                if isinstance(definition, dict) and "default" in definition:
                    defaults[key] = definition["default"]
                else:
                    defaults[key] = definition
            row["config_json"] = defaults
    return rows


def insertar_evento(
    camara_id: int,
    modulo: str,
    tipo_evento: str,
    duracion_seg: int,
    datos: dict,
    imagen_path: str = None,
) -> int:
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO eventos_ia
               (fk_camara, modulo, tipo_evento, duracion_seg, datos_json, imagen_path)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (camara_id, modulo, tipo_evento, duracion_seg, json.dumps(datos), imagen_path),
        )
        evento_id = conn.insert_id()
    conn.close()
    return evento_id


def insertar_snapshot(
    camara_id: int,
    personas: int,
    objetos: dict,
    mesas_sucias: int,
) -> None:
    """Inserta un snapshot de actividad por minuto para el Chat IA."""
    if personas == 0:
        nivel = "bajo"
    elif personas <= 5:
        nivel = "normal"
    elif personas <= 15:
        nivel = "alto"
    else:
        nivel = "critico"
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO snapshots_ia (fk_camara, personas, objetos_json, mesas_sucias, nivel)
               VALUES (%s, %s, %s, %s, %s)""",
            (camara_id, personas, json.dumps(objetos), mesas_sucias, nivel),
        )
    conn.close()


def get_snapshots(camara_id: int, limite: int = 20) -> list:
    """Retorna los ultimos N snapshots de una camara, del mas reciente al mas antiguo."""
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            """SELECT * FROM snapshots_ia
               WHERE fk_camara = %s
               ORDER BY rowid DESC LIMIT %s""",
            (camara_id, limite),
        )
        rows = cur.fetchall()
    conn.close()
    return rows


def marcar_alerta_enviada(evento_id: int):
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE eventos_ia SET alerta_enviada = 1 WHERE rowid = %s", (evento_id,)
        )
    conn.close()


def get_legos_activos() -> list:
    """Retorna todos los Legos activos con embedding y colores para el tracker."""
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT rowid, nombre, tipo, zona, puntos,
                   foto, metodo_reconocimiento, embedding_facial,
                   color_camiseta, color_pantalon, color_calzado,
                   color_piel, estilo_cabello, color_cabello, color_ojos,
                   complexion, altura
            FROM legos WHERE activo = 1
        """)
        rows = cur.fetchall()
    conn.close()
    for row in rows:
        if row.get("embedding_facial") and isinstance(row["embedding_facial"], str):
            try:
                row["embedding_facial"] = json.loads(row["embedding_facial"])
            except Exception:
                row["embedding_facial"] = None
    return rows


def actualizar_embedding(lego_id: int, embedding: list) -> None:
    """Guarda el vector facial (JSON) del Lego en BD."""
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE legos SET embedding_facial = %s WHERE rowid = %s",
            (json.dumps(embedding), lego_id),
        )
    conn.close()


def ajustar_puntos_lego(lego_id: int, delta: int, motivo: str, fk_evento: int = None) -> int:
    """Suma/resta puntos al Lego (clamped 0-1000) y registra el movimiento.
    Retorna el nuevo total de puntos."""
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE legos SET puntos = GREATEST(0, LEAST(1000, puntos + %s)) WHERE rowid = %s",
            (delta, lego_id),
        )
        cur.execute(
            "INSERT INTO lego_puntos_historial (fk_lego, puntos, motivo, fk_evento_ia) VALUES (%s, %s, %s, %s)",
            (lego_id, delta, motivo, fk_evento),
        )
        cur.execute("SELECT puntos FROM legos WHERE rowid = %s", (lego_id,))
        row = cur.fetchone()
    conn.close()
    return row["puntos"] if row else 0


def get_eventos(limite: int = 50, camara_id: int = None) -> list:
    conn = get_connection()
    sql = """SELECT e.*, c.nombre as camara_nombre
             FROM eventos_ia e
             LEFT JOIN camaras c ON e.fk_camara = c.rowid
             WHERE 1"""
    params = []
    if camara_id:
        sql += " AND e.fk_camara = %s"
        params.append(camara_id)
    sql += " ORDER BY e.rowid DESC LIMIT %s"
    params.append(limite)
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    conn.close()
    return rows


# ── Vision IA (FastVLM) ───────────────────────────────────────────────────

def insertar_vision_log(camara_id: int, modelo: str, descripcion: str) -> int:
    """Guarda una descripcion generada por Vision IA."""
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO vision_ia_log (fk_camara, modelo, descripcion) VALUES (%s, %s, %s)",
            (camara_id, modelo, descripcion),
        )
        rid = conn.insert_id()
    conn.close()
    return rid


def get_vision_log(camara_id: int, limite: int = 30) -> list:
    """Retorna las ultimas N descripciones de Vision IA de una camara."""
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            """SELECT rowid, fk_camara, modelo, descripcion, creado_fecha
               FROM vision_ia_log
               WHERE fk_camara = %s ORDER BY rowid DESC LIMIT %s""",
            (camara_id, limite),
        )
        rows = cur.fetchall()
    conn.close()
    return rows
