import cv2
import os
from ultralytics import YOLO
from processors.base_processor import BaseProcessor

class FaceBlurProcessor(BaseProcessor):
    """
    Detecta exclusivamente ROSTROS usando un modelo YOLO especializado.
    Ignora a las personas de espaldas y dibuja el difuminado exactamente sobre la cara.
    """

    def __init__(self, camara_id, config):
        super().__init__(camara_id, config)
        
        # Leemos exactamente el archivo que me mostraste en tu captura
        ruta_modelo = "models/yolov8n-face-lindevs.pt"
        
        if not os.path.exists(ruta_modelo):
            print(f"[FaceBlur - Cam {self.camara_id}] 🔴 ERROR: No se encontró {ruta_modelo}.")
            self.face_model = None
        else:
            print(f"[FaceBlur - Cam {self.camara_id}] Cargando modelo de rostros...")
            self.face_model = YOLO(ruta_modelo)
            
            # Enviar el modelo de rostros a la GPU (NVIDIA A2)
            try:
                import torch
                dev_cfg = os.getenv("IA_DEVICE", "auto").lower()
                dev = "cuda" if dev_cfg == "cuda" or (dev_cfg == "auto" and torch.cuda.is_available()) else "cpu"
                self.face_model.to(dev)
                print(f"[FaceBlur - Cam {self.camara_id}] 🟢 Modelo de rostros cargado en {dev}")
            except Exception as e:
                print(f"[FaceBlur - Cam {self.camara_id}] Aviso device Face YOLO: {e}")

    def procesar(self, frame, resultados):
        # Si el modelo no cargó, devolvemos el frame limpio
        if not self.face_model:
            return frame

        # Ejecutamos el modelo especialista en rostros
        # conf=0.3 asegura que detecte rostros claros sin agarrar basura del fondo
        resultados_caras = self.face_model(frame, verbose=False, conf=0.3)
        cajas_caras = resultados_caras[0].boxes

        for box in cajas_caras:
            # Obtener las coordenadas exactas de la cara detectada
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Validar límites de la imagen por seguridad matemática
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            # Extraer el cuadro exacto de la cara
            rostro_roi = frame[y1:y2, x1:x2]
            
            # Solo difuminar si el cuadro tiene dimensiones válidas
            if rostro_roi.shape[0] > 0 and rostro_roi.shape[1] > 0:
                # Aplicar blur fuerte
                rostro_blur = cv2.GaussianBlur(rostro_roi, (51, 51), 30)
                # Reemplazar los píxeles originales con los difuminados
                frame[y1:y2, x1:x2] = rostro_blur

        return frame