import os
import cv2
import time
import math
from ultralytics import YOLO
from processors.base_processor import BaseProcessor


class ErgonomicsTrackerProcessor(BaseProcessor):
    """
    Evalúa la sobrecarga física (Muri) usando YOLO-Pose.
    Detecta el esqueleto y calcula ángulos de inclinación de la espalda 
    para alertar sobre posturas forzadas.
    """

    def __init__(self, camara_id, config):
        super().__init__(camara_id, config)
        
        # Umbral de inclinación en grados (Ej: > 35 grados = postura forzada)
        self.umbral_inclinacion = int(self.config_extra.get("umbral_inclinacion", 35))
        
        # Cargar modelo Pose (Nano para que vuele en FPS)
        ruta_modelo = "models/yolo11n-pose.pt"
        
        if not os.path.exists(ruta_modelo):
            print(f"[Ergonomics - Cam {self.camara_id}] 🔴 ERROR: No se encontró {ruta_modelo}. Descárgalo en /models.")
            self.pose_model = None
        else:
            print(f"[Ergonomics - Cam {self.camara_id}] Cargando modelo Pose...")
            self.pose_model = YOLO(ruta_modelo)
            
            # Enviar a GPU si está disponible
            try:
                import torch
                dev_cfg = os.getenv("IA_DEVICE", "auto").lower()
                dev = "cuda" if dev_cfg == "cuda" or (dev_cfg == "auto" and torch.cuda.is_available()) else "cpu"
                self.pose_model.to(dev)
                print(f"[Ergonomics - Cam {self.camara_id}] 🟢 Modelo Pose cargado en {dev}")
            except Exception as e:
                print(f"[Ergonomics - Cam {self.camara_id}] Aviso device Pose YOLO: {e}")

    def _calcular_angulo_espalda(self, keypoints):
        """
        Recibe los 17 puntos del cuerpo.
        Calcula el ángulo de inclinación del torso respecto a la vertical.
        """
        # Índices en YOLO: 5=HombroIzq, 6=HombroDer, 11=CaderaIzq, 12=CaderaDer
        try:
            # Validar que los puntos existan y tengan confianza > 0.5
            puntos = keypoints.data[0] # Tensor de (17, 3) -> [x, y, conf]
            
            h_izq = puntos[5]
            h_der = puntos[6]
            c_izq = puntos[11]
            c_der = puntos[12]

            # Si la IA no ve claramente los hombros o caderas, no calculamos
            if h_izq[2] < 0.5 or h_der[2] < 0.5 or c_izq[2] < 0.5 or c_der[2] < 0.5:
                return None

            # 1. Encontrar el centro de los hombros (Cuello)
            cuello_x = (h_izq[0] + h_der[0]) / 2
            cuello_y = (h_izq[1] + h_der[1]) / 2

            # 2. Encontrar el centro de las caderas (Pelvis)
            pelvis_x = (c_izq[0] + c_der[0]) / 2
            pelvis_y = (c_izq[1] + c_der[1]) / 2

            # 3. Trigonometría: Calcular ángulo respecto a la línea vertical (Y)
            # dx = diferencia en ancho, dy = diferencia en alto
            dx = abs(cuello_x - pelvis_x)
            dy = abs(cuello_y - pelvis_y)
            
            # math.atan2 devuelve radianes, lo pasamos a grados
            angulo_rad = math.atan2(dx, dy) 
            angulo_grados = math.degrees(angulo_rad)
            
            return int(angulo_grados)
            
        except Exception:
            return None

    def procesar(self, frame, resultados):
        if not self.pose_model:
            return frame

        # Ejecutamos el modelo de pose en el frame actual
        res_pose = self.pose_model(frame, verbose=False, conf=0.5)
        
        if not res_pose or not res_pose[0].keypoints:
            return frame

        # --- LA MAGIA VISUAL AQUÍ ---
        # plot() dibuja el esqueleto automáticamente. 
        # boxes=False y labels=False evitan que dibuje el cuadro y el texto "person" por defecto.
        frame = res_pose[0].plot(labels=False, boxes=False)

        # Iterar sobre cada persona detectada para nuestros cálculos y textos
        for i, keypoints in enumerate(res_pose[0].keypoints):
            angulo = self._calcular_angulo_espalda(keypoints)
            
            if angulo is not None:
                # Obtener la caja para saber dónde dibujar nuestro texto
                box = res_pose[0].boxes[i]
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                
                # Regla Lean Muri: Evaluar si el ángulo es crítico
                if angulo >= self.umbral_inclinacion:
                    color = (0, 0, 255) # Rojo (Peligro Ergonómico)
                    alerta = f"POSTURA CRITICA: {angulo} grados"
                    
                    # Generar evento en BD (limitado por cooldown de 30s)
                    self._emitir_evento("sobrecarga_ergonomica", angulo, {
                        "alerta": "espalda_inclinada",
                        "angulo_detectado": angulo,
                        "umbral": self.umbral_inclinacion
                    })
                else:
                    color = (0, 255, 0) # Verde (Postura OK)
                    alerta = f"Postura OK: {angulo} grados"

                # Dibujar nuestro texto personalizado encima de la cabeza
                self._texto(frame, alerta, (bx1, max(by1 - 10, 20)), color, 0.6)

        return frame