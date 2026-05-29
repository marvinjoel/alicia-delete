import time
import cv2
from processors.base_processor import BaseProcessor


class StaffTrackerProcessor(BaseProcessor):
    """
    Rastrea al personal en el salon.
    Detecta si un mesero lleva demasiado tiempo en la misma zona sin moverse
    o si hay zonas de mesas sin presencia de personal.
    """

    def __init__(self, camara_id, config):
        super().__init__(camara_id, config)
        # zona_id -> {"ultima_vez": float, "conteo": int}
        self.historial_zonas = {}

    def procesar(self, frame, resultados):
        nombres = resultados[0].names
        cajas   = resultados[0].boxes
        ahora   = time.time()

        personal = [
            b for b in cajas
            if nombres[int(b.cls[0])] == "person" and float(b.conf[0]) > 0.5
        ]

        zonas_activas = set()

        for box in personal:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            zona_id = f"{x1 // 150}_{y1 // 150}"
            zonas_activas.add(zona_id)

            if zona_id not in self.historial_zonas:
                self.historial_zonas[zona_id] = {"ultima_vez": ahora, "conteo": 1}
            else:
                self.historial_zonas[zona_id]["ultima_vez"] = ahora
                self.historial_zonas[zona_id]["conteo"] += 1

        # Detectar personal inactivo (mismo lugar por mucho tiempo)
        zona_inactiva_peor = None
        inactivo_peor_seg  = 0
        for zona_id, info in self.historial_zonas.items():
            tiempo_zona = ahora - info["ultima_vez"]
            if tiempo_zona < 2 and info["conteo"] > 5:
                inactivo_seg = info["conteo"] * 0.1  # estimacion simple
                if inactivo_seg >= self.umbral_seg:
                    self._emitir_evento("mesero_inactivo", int(inactivo_seg), {
                        "zona": zona_id,
                    })
                if inactivo_seg > inactivo_peor_seg:
                    inactivo_peor_seg = inactivo_seg
                    zona_inactiva_peor = zona_id

        # Panel informativo
        self._panel(
            frame,
            f"Personal visible: {len(personal)}",
            x=10, y=90,
            color_fondo=(60, 60, 150),
        )

        # ── Log de progreso cada 3 segundos ───────────────────────────────
        if self._puede_loguear():
            umbral_fmt = self._fmt_tiempo(self.umbral_seg)
            if zona_inactiva_peor:
                dur_fmt = self._fmt_tiempo(inactivo_peor_seg)
                if inactivo_peor_seg >= self.umbral_seg:
                    print(f"[staff_tracker] Cam={self.camara_id} | "
                          f"Personal: {len(personal)} | INACTIVO zona={zona_inactiva_peor} | "
                          f"tiempo: {dur_fmt} / umbral: {umbral_fmt} → UMBRAL ALCANZADO")
                else:
                    faltan = self.umbral_seg - inactivo_peor_seg
                    print(f"[staff_tracker] Cam={self.camara_id} | "
                          f"Personal: {len(personal)} | INACTIVO zona={zona_inactiva_peor} | "
                          f"tiempo: {dur_fmt} / umbral: {umbral_fmt} → faltan {self._fmt_tiempo(faltan)}")
            else:
                print(f"[staff_tracker] Cam={self.camara_id} | "
                      f"OK — Personal visible: {len(personal)}")

        # Limpiar zonas antiguas
        obsoletas = [
            z for z, v in self.historial_zonas.items()
            if ahora - v["ultima_vez"] > 30
        ]
        for z in obsoletas:
            del self.historial_zonas[z]

        return frame
