import cv2
from processors.base_processor import BaseProcessor


class FaceBlurProcessor(BaseProcessor):
    """
    Detecta rostros (frontales y de perfil) usando OpenCV y aplica un filtro
    de desenfoque (blur). Optimizado para cámaras de seguridad (CCTV).
    """

    def __init__(self, camara_id, config):
        super().__init__(camara_id, config)
        
        try:
            # 1. Cargar modelo para rostros de frente
            ruta_frontal = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(ruta_frontal)
            
            # 2. Cargar modelo para rostros de perfil (muy común en la calle)
            ruta_perfil = cv2.data.haarcascades + 'haarcascade_profileface.xml'
            self.profile_cascade = cv2.CascadeClassifier(ruta_perfil)
            
            print(f"[FaceBlur - Cam {self.camara_id}] 🟢 Módulo inicializado. Modelos Haar cargados correctamente.")
        except Exception as e:
            print(f"[FaceBlur - Cam {self.camara_id}] 🔴 ERROR crítico al cargar modelos OpenCV: {e}")

    def procesar(self, frame, resultados):
        try:
            # Convertir a grises
            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detección Frontal (scaleFactor 1.05 es más lento pero más preciso, minSize más pequeño)
            rostros_frontales = self.face_cascade.detectMultiScale(
                gris, 
                scaleFactor=1.05, 
                minNeighbors=4, 
                minSize=(15, 15)
            )

            # Detección de Perfil
            rostros_perfil = self.profile_cascade.detectMultiScale(
                gris, 
                scaleFactor=1.05, 
                minNeighbors=4, 
                minSize=(15, 15)
            )

            # Consolidar todos los rostros detectados en una sola lista
            rostros = []
            if len(rostros_frontales) > 0:
                rostros.extend(rostros_frontales)
            if len(rostros_perfil) > 0:
                rostros.extend(rostros_perfil)

            # LOG DE TELEMETRÍA: Avisar a la consola solo si detectó algo
            if len(rostros) > 0:
                print(f"[FaceBlur - Cam {self.camara_id}] 👤 Detectados {len(rostros)} rostros en el frame actual.")

            # Aplicar desenfoque gaussiano a cada rostro
            for (x, y, w, h) in rostros:
                # Extraer ROI y aplicar blur
                rostro_roi = frame[y:y+h, x:x+w]
                rostro_blur = cv2.GaussianBlur(rostro_roi, (99, 99), 30)
                frame[y:y+h, x:x+w] = rostro_blur

        except Exception as e:
            # LOG DE ERROR: Evita que el worker muera si hay un fallo matemático
            print(f"[FaceBlur - Cam {self.camara_id}] ⚠️ Error procesando frame: {e}")

        return frame