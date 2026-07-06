import cv2
import time
import pytesseract
from processors.base_processor import BaseProcessor

class FormatChangeTrackerProcessor(BaseProcessor):
    """
    Detecta cambios de formato en línea de jabón (ej: 100g -> 120g) usando OCR.
    Hereda de BaseProcessor para integración Zero-Copy con CameraWorker.
    """

    def __init__(self, camara_id, config):
        super().__init__(camara_id, config)
        
        # Estado del formato
        self.last_format = None
        self.change_start_time = None
        self.current_turno = 'A'  # Por defecto. Podría leerse de la BD en un futuro.
        
        # Coordenadas de la pantalla en la cámara (ROI: x1, y1, x2, y2)
        # Se pueden ajustar desde el panel web vía config_json
        self.roi = self.config_extra.get("roi_pantalla", [50, 200, 250, 300])
        
        # Lista de formatos válidos a buscar para evitar falsos positivos
        self.formatos_validos = ['100g', '120g', '150g', '200g', '250g']
        
        # Optimización: No hacer OCR en cada frame (quemaría la CPU). Haremos OCR cada 2 segundos.
        self.last_ocr_time = 0
        self.ocr_interval = 2.0 

    def _extraer_texto(self, frame):
        """Aplica OCR sobre la Región de Interés (ROI)."""
        x1, y1, x2, y2 = self.roi
        
        # Validar límites para evitar crasheos si el ROI sale del frame
        alto, ancho = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(ancho, x2), min(alto, y2)
        
        roi_img = frame[y1:y2, x1:x2]
        
        if roi_img.size == 0:
            return None

        # Preprocesamiento para mejorar el OCR
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Configuración Tesseract: psm 6 (bloque de texto uniforme)
        custom_config = r'--oem 3 --psm 6'
        try:
            texto = pytesseract.image_to_string(enhanced, config=custom_config).lower()
            
            # Buscar si el texto contiene alguno de los formatos válidos
            for fmt in self.formatos_validos:
                if fmt in texto:
                    return fmt
        except Exception as e:
            print(f"[{self.modulo}] Cam={self.camara_id} | Error OCR: {e}")
            
        return None

    def procesar(self, frame, resultados):
        ahora = time.time()
        x1, y1, x2, y2 = self.roi

        # Dibujar la zona donde la IA está leyendo (ROI)
        self._caja(frame, x1, y1, x2, y2, color=(255, 100, 0), grosor=2)
        
        # Frame Skipping: Solo procesamos el OCR cada N segundos
        if ahora - self.last_ocr_time >= self.ocr_interval:
            self.last_ocr_time = ahora
            formato_detectado = self._extraer_texto(frame)
            
            if formato_detectado:
                # Si detectamos un cambio de formato (y no es el primer frame)
                if formato_detectado != self.last_format:
                    if self.last_format is not None:
                        # Calcular duración del cambio
                        duracion_sec = int(ahora - self.change_start_time)
                        
                        # Evitar ruido: solo registrar si duró más de 10 segundos
                        if duracion_sec >= 10:
                            import datetime
                            datos_evento = {
                                "formato_origen": self.last_format,
                                "formato_destino": formato_detectado,
                                "duracion_sec": duracion_sec,
                                "turno": self.current_turno,
                                "inicio_datetime": datetime.datetime.fromtimestamp(self.change_start_time).isoformat()
                            }
                            
                            # Insertar en la BD y notificar por WebSocket
                            self._emitir_evento("cambio_formato", duracion_sec, datos_evento)
                            print(f"[{self.modulo}] Cam={self.camara_id} | CAMBIO REGISTRADO: {self.last_format} -> {formato_detectado} ({duracion_sec}s)")

                    # Iniciar el nuevo ciclo
                    self.last_format = formato_detectado
                    self.change_start_time = ahora

        # Mostrar el formato actual en pantalla en tiempo real
        if self.last_format:
            tiempo_transcurrido = int(ahora - self.change_start_time) if self.change_start_time else 0
            self._panel(frame, f"Formato: {self.last_format.upper()} ({self._fmt_tiempo(tiempo_transcurrido)})", x1, max(y1 - 40, 10), color_fondo=(200, 100, 0))
        else:
            self._panel(frame, "Formato: LEYENDO...", x1, max(y1 - 40, 10), color_fondo=(100, 100, 100))

        return frame