import time
import cv2
from processors.base_processor import BaseProcessor


class PeopleCounterProcessor(BaseProcessor):
    """
    Cuenta personas en el local y alerta si se supera la capacidad maxima.
    La capacidad maxima se configura en config_json: {"capacidad_max": 40}
    La alerta (WhatsApp) se dispara cuando el exceso supera umbral_minutos en BD.
    """

    def __init__(self, camara_id, config):
        super().__init__(camara_id, config)
        if "capacidad_max" not in self.config_extra:
            raise ValueError("Falta capacidad_max en config_json para people_counter")
        self.capacidad_max    = int(self.config_extra.get("capacidad_max", 0))
        self.inicio_exceso    = None  # timer: cuando se empezo a superar la capacidad
        self.inicio_presencia = None  # timer: cuando aparecio la primera persona (estado OK)
        # _ultimo_log y cooldown de BD heredados de BaseProcessor
        print(f"[people_counter] Cam={camara_id} | capacidad_max={self.capacidad_max} | umbral={self.umbral_seg}s")

    def procesar(self, frame, resultados):
        nombres = resultados[0].names
        cajas   = resultados[0].boxes
        ahora   = time.time()

        personas = [
            b for b in cajas
            if nombres[int(b.cls[0])] == "person" and float(b.conf[0]) > 0.5
        ]
        conteo = len(personas)
        excede = conteo > self.capacidad_max
        umbral_fmt = self._fmt_tiempo(self.umbral_seg)

        if excede:
            self.inicio_presencia = None  # resetear timer OK
            if self.inicio_exceso is None:
                self.inicio_exceso = ahora
            duracion     = ahora - self.inicio_exceso
            duracion_int = int(duracion)
            dur_fmt      = self._fmt_tiempo(duracion)

            # ── Log de progreso cada 3 segundos ───────────────────────────
            if self._puede_loguear():
                if self.umbral_seg == 0:
                    print(f"[people_counter] Cam={self.camara_id} | "
                          f"EXCEDE {conteo}>{self.capacidad_max} | "
                          f"umbral=0 min → EVENTO INMEDIATO")
                elif duracion >= self.umbral_seg:
                    print(f"[people_counter] Cam={self.camara_id} | "
                          f"EXCEDE {conteo}>{self.capacidad_max} | "
                          f"tiempo: {dur_fmt} / umbral: {umbral_fmt} → UMBRAL ALCANZADO")
                else:
                    faltan = self.umbral_seg - duracion
                    print(f"[people_counter] Cam={self.camara_id} | "
                          f"EXCEDE {conteo}>{self.capacidad_max} | "
                          f"tiempo: {dur_fmt} / umbral: {umbral_fmt} → faltan {self._fmt_tiempo(faltan)}")

            # ── Paneles en el frame ────────────────────────────────────────
            self._panel(frame, f"Personas: {conteo} / {self.capacidad_max}",
                        x=10, y=10, color_fondo=(0, 0, 180))
            self._panel(frame, f"Aforo excedido  {dur_fmt}",
                        x=10, y=50, color_fondo=(0, 0, 160))

            # ── Emitir evento solo cuando se alcanza el umbral ─────────────
            # umbral_seg=0 → emitir desde el primer frame de exceso
            if duracion >= self.umbral_seg:
                self._emitir_evento("aforo_excedido", duracion_int, {
                    "conteo":        conteo,
                    "capacidad_max": self.capacidad_max,
                })

        else:
            self.inicio_exceso = None
            if conteo > 0:
                # Hay personas pero sin exceder: arrancar timer de presencia
                if self.inicio_presencia is None:
                    self.inicio_presencia = ahora
                duracion = ahora - self.inicio_presencia
                dur_fmt  = self._fmt_tiempo(duracion)

                if self._puede_loguear():
                    if self.umbral_seg == 0:
                        print(f"[people_counter] Cam={self.camara_id} | "
                              f"OK ({conteo}<={self.capacidad_max}) | "
                              f"tiempo: {dur_fmt} / umbral: {umbral_fmt}")
                    else:
                        faltan = max(0, self.umbral_seg - duracion)
                        print(f"[people_counter] Cam={self.camara_id} | "
                              f"OK ({conteo}<={self.capacidad_max}) | "
                              f"tiempo: {dur_fmt} / umbral: {umbral_fmt} → faltan {self._fmt_tiempo(faltan)}")
            else:
                # Sin personas: resetear timer y loguear sin contador
                self.inicio_presencia = None
                if self._puede_loguear():
                    print(f"[people_counter] Cam={self.camara_id} | "
                          f"OK — sin personas detectadas | umbral: {umbral_fmt}")

            self._panel(frame, f"Personas: {conteo} / {self.capacidad_max}",
                        x=10, y=10, color_fondo=(0, 120, 0))

        return frame
