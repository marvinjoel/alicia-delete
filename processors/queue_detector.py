import time
import cv2
from processors.base_processor import BaseProcessor


class QueueDetectorProcessor(BaseProcessor):
    """
    Detecta cola de espera en la entrada del restaurante.
    Cuenta personas de pie en la zona de entrada (primer tercio del frame).
    Alerta si la cola supera N personas por mas del umbral configurado.
    """

    def __init__(self, camara_id, config):
        super().__init__(camara_id, config)
        self.cola_minima = int(self.config_extra.get("cola_minima", 3))
        self.inicio_cola = None

    def procesar(self, frame, resultados):
        nombres = resultados[0].names
        cajas   = resultados[0].boxes
        ahora   = time.time()

        alto, ancho = frame.shape[:2]
        # Zona de entrada: primer tercio izquierdo del frame
        limite_x = ancho // 3

        personas_entrada = []
        for box in cajas:
            if nombres[int(box.cls[0])] != "person" or float(box.conf[0]) < 0.5:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            centro_x = (x1 + x2) // 2
            if centro_x <= limite_x:
                personas_entrada.append((x1, y1, x2, y2))

        conteo = len(personas_entrada)

        # Dibujar linea divisoria de zona de entrada
        cv2.line(frame, (limite_x, 0), (limite_x, alto), (255, 200, 0), 1)

        if conteo >= self.cola_minima:
            if self.inicio_cola is None:
                self.inicio_cola = ahora
            duracion   = ahora - self.inicio_cola
            dur_fmt    = self._fmt_tiempo(duracion)
            umbral_fmt = self._fmt_tiempo(self.umbral_seg)

            color = (0, 180, 255) if duracion < self.umbral_seg else (0, 0, 220)
            self._panel(
                frame,
                f"Cola entrada: {conteo} personas  {dur_fmt}",
                x=10, y=130,
                color_fondo=(120, 80, 0),
            )

            if duracion >= self.umbral_seg:
                self._emitir_evento("cola_entrada", int(duracion), {
                    "conteo": conteo,
                    "cola_minima": self.cola_minima,
                })

            # ── Log de progreso cada 3 segundos ───────────────────────────
            if self._puede_loguear():
                if self.umbral_seg == 0:
                    print(f"[queue_detector] Cam={self.camara_id} | "
                          f"COLA {conteo} personas (min={self.cola_minima}) | "
                          f"umbral=0 → EVENTO INMEDIATO")
                elif duracion >= self.umbral_seg:
                    print(f"[queue_detector] Cam={self.camara_id} | "
                          f"COLA {conteo} personas (min={self.cola_minima}) | "
                          f"tiempo: {dur_fmt} / umbral: {umbral_fmt} → UMBRAL ALCANZADO")
                else:
                    faltan = self.umbral_seg - duracion
                    print(f"[queue_detector] Cam={self.camara_id} | "
                          f"COLA {conteo} personas (min={self.cola_minima}) | "
                          f"tiempo: {dur_fmt} / umbral: {umbral_fmt} → faltan {self._fmt_tiempo(faltan)}")
        else:
            self.inicio_cola = None
            if self._puede_loguear():
                print(f"[queue_detector] Cam={self.camara_id} | "
                      f"OK — cola: {conteo} < minimo: {self.cola_minima}")

        return frame
