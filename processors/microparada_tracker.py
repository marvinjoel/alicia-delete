import time
import cv2
from processors.base_processor import BaseProcessor

class MicroparadaTrackerProcessor(BaseProcessor):
    """
    Detecta microparadas en la línea de producción (detenciones > 30 segundos).
    Hereda de BaseProcessor para integración Zero-Copy.
    """

    def __init__(self, camara_id, config):
        super().__init__(camara_id, config)
        
        # Parámetros desde config_json o por defecto
        self.vel_threshold = int(self.config_extra.get("vel_threshold", 10))
        self.duration_threshold_sec = int(self.config_extra.get("duration_threshold", 30))
        self.estacion = self.config_extra.get("estacion", "Moldeadora")
        
        # Estado interno
        self.parada_start = None
        self.last_velocity = 15  # Simulamos una velocidad normal al iniciar (ej. 15 bpm)
        self.current_turno = 'A'

    def _estimar_velocidad_visual(self, resultados):
        """
        MVP: Heurística visual para estimar si la línea está operando.
        En producción real, esto se conectaría al procesador WIP o a un PLC.
        Aquí asumimos que si no hay operarios en el frame, la línea está parada.
        """
        nombres = resultados[0].names
        cajas = resultados[0].boxes
        
        # Contar personas (operarios) en el frame
        operarios = [b for b in cajas if nombres[int(b.cls[0])] == "person" and float(b.conf[0]) > 0.4]
        
        # Si no hay operarios en la estación, la velocidad es 0. 
        # Si hay operarios, simulamos velocidad normal (15 bpm).
        return 0 if len(operarios) == 0 else 15

    def _inferir_causa(self, resultados):
        """
        Infiere la causa de la microparada. 
        Para el MVP usamos una regla simple, el Backend Team puede sumar YOLOv8-jamming luego.
        """
        # Causas Pareto: 'Atasco de material', 'Falta de material', 'Ajuste mecánico', etc.
        return "Falta de personal / Ajuste mecánico"

    def procesar(self, frame, resultados):
        ahora = time.time()
        
        # 1. Obtener velocidad actual (simulada por heurística visual)
        velocidad = self._estimar_velocidad_visual(resultados)
        self.last_velocity = velocidad

        # 2. Máquina de estados para la detención
        if velocidad < self.vel_threshold:
            # La línea se detuvo. Iniciar o mantener cronómetro.
            if self.parada_start is None:
                self.parada_start = ahora
            
            duracion_parada = ahora - self.parada_start
            
            # Dibujar UI de Alerta
            if duracion_parada > self.duration_threshold_sec:
                color_bg = (0, 0, 220)  # Rojo crítico
                texto = f"MICROPARADA CONFIRMADA: {self._fmt_tiempo(duracion_parada)}"
                
                # Emitir evento a la BD (BaseProcessor maneja el cooldown para no hacer spam)
                causa = self._inferir_causa(resultados)
                datos_evento = {
                    "causa": causa,
                    "estacion": self.estacion,
                    "duracion_seg": int(duracion_parada),
                    "turno": self.current_turno,
                    "velocidad_bpm": velocidad
                }
                self._emitir_evento("microparada", int(duracion_parada), datos_evento)
                
            else:
                color_bg = (0, 140, 255)  # Naranja advirtiendo
                texto = f"Advertencia - Baja Vel: {self._fmt_tiempo(duracion_parada)}"
                
            self._panel(frame, texto, 10, 50, color_fondo=color_bg)
            
        else:
            # La línea está en movimiento (Velocidad >= umbral)
            if self.parada_start is not None:
                # Se restableció la línea. Reiniciar cronómetro.
                duracion_total = ahora - self.parada_start
                if duracion_total >= self.duration_threshold_sec:
                    print(f"[{self.modulo}] Cam={self.camara_id} | FIN DE MICROPARADA. Duración total: {int(duracion_total)}s")
                self.parada_start = None
                
            self._panel(frame, f"Linea OK: {velocidad} bpm", 10, 50, color_fondo=(0, 150, 0))

        return frame