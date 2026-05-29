import os
import time
import numpy as np
import cv2
from processors.base_processor import BaseProcessor

# deepface es opcional — si no esta instalado o falla solo funciona vestimenta
try:
    from deepface import DeepFace
    DEEPFACE_OK = True
except Exception as _e:
    DEEPFACE_OK = False
    print(f"[lego_tracker] AVISO: deepface no disponible ({_e}). "
          "Instala con: pip install deepface tf-keras")


# ── Helpers de color ───────────────────────────────────────────────────────────

def _hex_a_hsv(hex_color: str) -> np.ndarray:
    """Convierte color #RRGGBB a array HSV numpy."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    bgr = np.uint8([[[b, g, r]]])
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0][0].astype(float)


def _color_dominante(region: np.ndarray) -> np.ndarray:
    """Extrae color dominante de una region BGR en espacio HSV."""
    if region is None or region.size == 0:
        return np.array([0.0, 0.0, 0.0])
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    # Filtrar pixels grises/blancos/negros (poca saturacion)
    mask = hsv[:, 1] > 30
    return np.mean(hsv[mask], axis=0) if mask.sum() > 10 else np.mean(hsv, axis=0)


def _dist_color(hsv1: np.ndarray, hsv2: np.ndarray) -> float:
    """Distancia normalizada entre dos colores HSV (0=igual, 1=opuesto)."""
    dh = min(abs(hsv1[0] - hsv2[0]), 180 - abs(hsv1[0] - hsv2[0])) / 90.0
    ds = abs(hsv1[1] - hsv2[1]) / 255.0
    dv = abs(hsv1[2] - hsv2[2]) / 255.0
    return dh * 0.6 + ds * 0.3 + dv * 0.1


def _similitud_coseno(a: list, b: list) -> float:
    """Similitud coseno entre dos vectores (1=identico, 0=no relacionado)."""
    va, vb = np.array(a, dtype=float), np.array(b, dtype=float)
    norma = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / norma) if norma > 0 else 0.0


# ── Procesador ─────────────────────────────────────────────────────────────────

class LegoTrackerProcessor(BaseProcessor):
    """
    Identifica personas en camara y las asocia con perfiles Lego.
    Metodos: 'facial' (deepface), 'vestimenta' (colores), 'ambos'.
    Configurable por Lego (metodo_reconocimiento) y por camara (config_json).
    """

    def __init__(self, camara_id, config):
        super().__init__(camara_id, config)
        # Override del metodo para toda la camara (sobreescribe el del Lego)
        self.metodo_override = self.config_extra.get("metodo_reconocimiento", None)
        # Puntos a ajustar al detectar (negativo = penalizar, positivo = bonificar)
        self.ajuste_puntos   = int(self.config_extra.get("ajuste_puntos", 0))
        # Umbrales de confianza
        self.umbral_facial   = float(self.config_extra.get("umbral_facial", 0.72))
        self.umbral_ropa     = float(self.config_extra.get("umbral_ropa",   0.55))

        # Cache de legos: se recarga cada 60s sin reiniciar el worker
        self._legos         = []
        self._ultimo_reload = 0.0

        # Cooldown de ajuste de puntos por Lego: 5 minutos para no spamear
        self._puntos_ts = {}   # lego_id -> timestamp ultimo ajuste
        self._puntos_cd = 300

        # Ruta base PHP para localizar fotos de perfil
        self._uploads_base = os.getenv("PHP_UPLOADS_PATH", r"d:\Instalaciones\laragon\www\alicia")

        print(f"[lego_tracker] Cam={camara_id} | "
              f"metodo_override={self.metodo_override} | ajuste_puntos={self.ajuste_puntos} | "
              f"umbral_facial={self.umbral_facial} | umbral_ropa={self.umbral_ropa}")

    # ── Carga / recarga de legos ───────────────────────────────────────────────

    def _recargar_legos(self):
        ahora = time.time()
        if ahora - self._ultimo_reload < 60:
            return
        from services.database import get_legos_activos
        try:
            self._legos = get_legos_activos()
            self._ultimo_reload = ahora
            print(f"[lego_tracker] Cam={self.camara_id} | Legos cargados: {len(self._legos)}")
        except Exception as e:
            print(f"[lego_tracker] Cam={self.camara_id} | ERROR cargando legos: {e}")

    # ── Match facial ──────────────────────────────────────────────────────────

    def _match_facial(self, face_crop: np.ndarray, lego: dict) -> float:
        if not DEEPFACE_OK:
            return 0.0
        emb_guardado = lego.get("embedding_facial")
        if not emb_guardado:
            return 0.0
        try:
            result = DeepFace.represent(
                img_path=face_crop,
                model_name="Facenet",
                enforce_detection=False,
            )
            return _similitud_coseno(result[0]["embedding"], emb_guardado)
        except Exception:
            return 0.0

    # ── Match vestimenta ──────────────────────────────────────────────────────

    def _match_vestimenta(self, frame: np.ndarray, x1, y1, x2, y2, lego: dict) -> float:
        alto = y2 - y1
        region_cam  = frame[y1 + int(alto * 0.25): y1 + int(alto * 0.55), x1:x2]
        region_pant = frame[y1 + int(alto * 0.55): y2,                     x1:x2]

        hsv_cam_det  = _color_dominante(region_cam)
        hsv_pant_det = _color_dominante(region_pant)

        try:
            hsv_cam_lego  = _hex_a_hsv(lego.get("color_camiseta", "#808080"))
            hsv_pant_lego = _hex_a_hsv(lego.get("color_pantalon", "#404040"))
        except Exception:
            return 0.0

        sim_cam  = max(0.0, 1.0 - _dist_color(hsv_cam_det,  hsv_cam_lego)  * 2)
        sim_pant = max(0.0, 1.0 - _dist_color(hsv_pant_det, hsv_pant_lego) * 2)
        return sim_cam * 0.6 + sim_pant * 0.4

    # ── Identificar persona ───────────────────────────────────────────────────

    def _identificar(self, frame, x1, y1, x2, y2):
        """Retorna (lego_match | None, confianza, metodo)."""
        alto      = y2 - y1
        face_crop = frame[y1: y1 + int(alto * 0.35), x1:x2]

        mejor_lego, mejor_conf, mejor_metodo = None, 0.0, ""

        for lego in self._legos:
            metodo = self.metodo_override or lego.get("metodo_reconocimiento", "facial")

            conf_facial = self._match_facial(face_crop, lego) if metodo in ("facial", "ambos") else 0.0
            conf_ropa   = self._match_vestimenta(frame, x1, y1, x2, y2, lego) if metodo in ("vestimenta", "ambos") else 0.0

            if metodo == "facial":
                conf, ok = conf_facial, conf_facial >= self.umbral_facial
            elif metodo == "vestimenta":
                conf, ok = conf_ropa,   conf_ropa   >= self.umbral_ropa
            else:  # ambos
                conf = (conf_facial + conf_ropa) / 2
                ok   = conf_facial >= self.umbral_facial and conf_ropa >= self.umbral_ropa

            print(f"[lego_tracker] DEBUG {lego['nombre']} | facial={conf_facial:.2f} ropa={conf_ropa:.2f} conf={conf:.2f} ok={ok}")
            if ok and conf > mejor_conf:
                mejor_lego, mejor_conf, mejor_metodo = lego, conf, metodo

        return mejor_lego, mejor_conf, mejor_metodo

    # ── Nivel segun puntos ────────────────────────────────────────────────────

    @staticmethod
    def _nivel(puntos: int) -> str:
        if puntos >= 80: return "Oro"
        if puntos >= 60: return "Plata"
        if puntos >= 35: return "Bronce"
        return "Alerta"

    # ── Loop principal ────────────────────────────────────────────────────────

    def procesar(self, frame, resultados):
        self._recargar_legos()
        if not self._legos:
            return frame

        nombres = resultados[0].names
        cajas   = resultados[0].boxes
        ahora   = time.time()

        personas = [
            b for b in cajas
            if nombres[int(b.cls[0])] == "person" and float(b.conf[0]) > 0.5
        ]

        identificados = 0

        for box in personas:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            lego, conf, metodo = self._identificar(frame, x1, y1, x2, y2)

            if lego:
                identificados += 1
                puntos = lego["puntos"]
                nivel  = self._nivel(puntos)
                color  = (0, 200, 0) if puntos >= 80 else (0, 180, 255) if puntos >= 40 else (0, 0, 220)

                # Dibujar sobre el frame
                self._caja(frame, x1, y1, x2, y2, color)
                self._texto(frame, f"{lego['nombre']}  {puntos}pts", (x1, max(y1 - 22, 18)), color)
                self._texto(frame, f"{int(conf * 100)}% {metodo}", (x1, max(y1 - 6, 30)), (180, 180, 180), 0.5)

                # Push WebSocket al dashboard
                self._push_ws("lego_detectado", 0, {
                    "lego_id":        lego["rowid"],
                    "nombre":         lego["nombre"],
                    "puntos":         puntos,
                    "nivel":          nivel,
                    "confianza":      round(conf, 2),
                    "metodo":         metodo,
                    "color_camiseta": lego.get("color_camiseta"),
                    "color_pantalon": lego.get("color_pantalon"),
                    "color_calzado":  lego.get("color_calzado"),
                    "color_piel":     lego.get("color_piel"),
                    "estilo_cabello": lego.get("estilo_cabello"),
                    "color_cabello":  lego.get("color_cabello"),
                    "color_ojos":     lego.get("color_ojos"),
                })

                # Ajuste de puntos con cooldown por Lego
                if self.ajuste_puntos != 0:
                    lego_id = lego["rowid"]
                    if ahora - self._puntos_ts.get(lego_id, 0) >= self._puntos_cd:
                        from services.database import ajustar_puntos_lego
                        nuevos = ajustar_puntos_lego(
                            lego_id,
                            self.ajuste_puntos,
                            f"lego_tracker cam={self.camara_id} metodo={metodo}",
                        )
                        self._puntos_ts[lego_id] = ahora
                        lego["puntos"] = nuevos  # actualizar cache local
                        print(f"[lego_tracker] Cam={self.camara_id} | "
                              f"{lego['nombre']}: {self.ajuste_puntos:+d}pts → total {nuevos}pts")
            else:
                # Persona no identificada
                self._caja(frame, x1, y1, x2, y2, (80, 80, 80))
                self._texto(frame, "?", (x1 + 4, max(y1 - 6, 18)), (140, 140, 140), 0.55)

        # Log cada 3 segundos
        if self._puede_loguear():
            print(f"[lego_tracker] Cam={self.camara_id} | "
                  f"Personas={len(personas)} | Identificados={identificados}/{len(personas)}")

        return frame
