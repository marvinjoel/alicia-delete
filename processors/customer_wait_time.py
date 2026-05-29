import time
from processors.base_processor import BaseProcessor


class CustomerWaitTimeProcessor(BaseProcessor):
    """
    Detecta clientes sentados que llevan mucho tiempo sin ser atendidos.
    Compara la presencia de personas sentadas vs meseros en la misma zona.
    Si hay cliente sin mesero cercano por mas del umbral, genera alerta.
    """

    def __init__(self, camara_id, config):
        super().__init__(camara_id, config)
        # zona_id -> {"inicio": float, "ultimo": float}
        self.esperas_activas = {}

    def procesar(self, frame, resultados):
        nombres = resultados[0].names
        cajas   = resultados[0].boxes
        ahora   = time.time()

        personas = [
            (int(b.xyxy[0][0]), int(b.xyxy[0][1]), int(b.xyxy[0][2]), int(b.xyxy[0][3]))
            for b in cajas
            if nombres[int(b.cls[0])] == "person" and float(b.conf[0]) > 0.5
        ]

        zonas_con_persona = set()

        for (x1, y1, x2, y2) in personas:
            zona_id = f"{x1 // 200}_{y1 // 200}"
            zonas_con_persona.add(zona_id)

            if zona_id not in self.esperas_activas:
                self.esperas_activas[zona_id] = {"inicio": ahora, "ultimo": ahora}
            else:
                self.esperas_activas[zona_id]["ultimo"] = ahora

            duracion = ahora - self.esperas_activas[zona_id]["inicio"]

            if duracion >= self.umbral_seg:
                color = (0, 0, 220)
                self._caja(frame, x1, y1, x2, y2, color)
                self._texto(
                    frame,
                    f"ESPERA {self._fmt_tiempo(duracion)}",
                    (x1, max(y1 - 8, 16)),
                    color,
                )
                self._emitir_evento("cliente_sin_atencion", int(duracion), {
                    "zona": zona_id,
                })

        # ── Log de progreso cada 3 segundos ───────────────────────────────
        if self._puede_loguear():
            umbral_fmt = self._fmt_tiempo(self.umbral_seg)
            if not self.esperas_activas:
                print(f"[customer_wait] Cam={self.camara_id} | OK — sin clientes en espera")
            else:
                # Zona con mayor tiempo de espera
                peor_zona, peor_dur = max(
                    ((z, ahora - v["inicio"]) for z, v in self.esperas_activas.items()),
                    key=lambda x: x[1],
                )
                dur_fmt = self._fmt_tiempo(peor_dur)
                n = len(self.esperas_activas)
                zonas_txt = f"{n} zona{'s' if n > 1 else ''}"
                if peor_dur >= self.umbral_seg:
                    print(f"[customer_wait] Cam={self.camara_id} | "
                          f"ESPERA {zonas_txt} | peor zona={peor_zona} | "
                          f"tiempo: {dur_fmt} / umbral: {umbral_fmt} → UMBRAL ALCANZADO")
                else:
                    faltan = self.umbral_seg - peor_dur
                    print(f"[customer_wait] Cam={self.camara_id} | "
                          f"ESPERA {zonas_txt} | peor zona={peor_zona} | "
                          f"tiempo: {dur_fmt} / umbral: {umbral_fmt} → faltan {self._fmt_tiempo(faltan)}")

        # Limpiar zonas sin persona
        inactivas = [
            z for z in self.esperas_activas
            if z not in zonas_con_persona
            and ahora - self.esperas_activas[z]["ultimo"] > 5
        ]
        for z in inactivas:
            del self.esperas_activas[z]

        return frame
