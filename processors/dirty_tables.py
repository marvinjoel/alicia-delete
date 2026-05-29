import time
from processors.base_processor import BaseProcessor

# Clases COCO que indican suciedad sobre una mesa
CLASES_SUCIEDAD = {
    "cup", "bowl", "bottle", "fork", "knife", "spoon",
    "wine glass", "plate", "dining table"
}


class DirtyTableProcessor(BaseProcessor):
    """
    Detecta mesas sucias buscando vajilla usada (platos, vasos, cubiertos).
    Cronometra desde la primera deteccion y alerta si supera el umbral.
    """

    def __init__(self, camara_id, config):
        super().__init__(camara_id, config)
        # zona_id -> {"inicio": float, "ultimo": float}
        self.zonas_activas = {}

    def procesar(self, frame, resultados):
        nombres = resultados[0].names
        cajas   = resultados[0].boxes
        ahora   = time.time()

        zonas_vistas = set()

        for box in cajas:
            clase_nom = nombres[int(box.cls[0])]
            confianza = float(box.conf[0])

            if clase_nom not in CLASES_SUCIEDAD or confianza < 0.45:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Identificar zona en cuadricula de 150px para agrupar objetos de la misma mesa
            zona_id = f"{x1 // 150}_{y1 // 150}"
            zonas_vistas.add(zona_id)

            if zona_id not in self.zonas_activas:
                self.zonas_activas[zona_id] = {"inicio": ahora, "ultimo": ahora}
            else:
                self.zonas_activas[zona_id]["ultimo"] = ahora

            duracion = ahora - self.zonas_activas[zona_id]["inicio"]
            color    = (0, 100, 255) if duracion < self.umbral_seg else (0, 0, 220)

            self._caja(frame, x1, y1, x2, y2, color)
            self._texto(
                frame,
                f"SUCIA {self._fmt_tiempo(duracion)}",
                (x1, max(y1 - 8, 16)),
                color,
            )

            if duracion >= self.umbral_seg:
                self._emitir_evento("mesa_sucia", int(duracion), {
                    "zona": zona_id,
                    "clase_detectada": clase_nom,
                    "confianza": round(confianza, 2),
                })

        # ── Log de progreso cada 3 segundos ───────────────────────────────
        if self._puede_loguear():
            umbral_fmt = self._fmt_tiempo(self.umbral_seg)
            if not self.zonas_activas:
                print(f"[dirty_tables] Cam={self.camara_id} | OK — sin vajilla detectada")
            else:
                # Mostrar la zona con mas tiempo (la mas critica)
                peor_zona, peor_dur = max(
                    ((z, ahora - v["inicio"]) for z, v in self.zonas_activas.items()),
                    key=lambda x: x[1],
                )
                dur_fmt = self._fmt_tiempo(peor_dur)
                n = len(self.zonas_activas)
                zonas_txt = f"{n} zona{'s' if n > 1 else ''}"
                if peor_dur >= self.umbral_seg:
                    print(f"[dirty_tables] Cam={self.camara_id} | "
                          f"SUCIA {zonas_txt} | peor zona={peor_zona} | "
                          f"tiempo: {dur_fmt} / umbral: {umbral_fmt} → UMBRAL ALCANZADO")
                else:
                    faltan = self.umbral_seg - peor_dur
                    print(f"[dirty_tables] Cam={self.camara_id} | "
                          f"SUCIA {zonas_txt} | peor zona={peor_zona} | "
                          f"tiempo: {dur_fmt} / umbral: {umbral_fmt} → faltan {self._fmt_tiempo(faltan)}")

        # Limpiar zonas sin actividad por mas de 8 segundos
        inactivas = [
            z for z, v in self.zonas_activas.items()
            if z not in zonas_vistas and ahora - v["ultimo"] > 8
        ]
        for z in inactivas:
            del self.zonas_activas[z]

        return frame
