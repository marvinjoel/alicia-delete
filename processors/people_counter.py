import cv2
from processors.base_processor import BaseProcessor

class PeopleCounterProcessor(BaseProcessor):
    """
    Procesador ultra-simple: cuenta personas y muestra el total en pantalla.
    Estilo visual High-Tech / Profesional para presentaciones.
    """

    def procesar(self, frame, resultados):
        nombres = resultados[0].names
        cajas   = resultados[0].boxes
        
        # Filtrar solo personas con confianza mayor a 0.5
        personas = [
            b for b in cajas 
            if nombres[int(b.cls[0])] == "person" and float(b.conf[0]) > 0.5
        ]
        
        conteo = len(personas)

        # 1. UI del Contador: Usamos tu método _panel para un fondo oscuro semitransparente
        # Se ve mucho más limpio que el texto flotante gigante.
        self._panel(
            frame, 
            f"Personas en escena: {conteo}", 
            x=20, y=30, 
            color_fondo=(40, 40, 40) # Fondo gris oscuro elegante
        )

        # 2. Estilo de las cajas (BGR format en OpenCV)
        # Un cian/azul claro da un aspecto de IA muy moderno y limpio
        color_caja = (255, 200, 50) 
        grosor_linea = 1 # Línea principal muy delgada

        for b in personas:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            
            # Dibujar la caja principal delgada
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_caja, grosor_linea)
            
            # --- DETALLE PROFESIONAL: Esquinas reforzadas (Estilo HUD/Visor) ---
            l = 12  # longitud de la esquinita
            t = 2   # grosor de la esquinita (ligeramente más grueso que la caja)
            
            # Esquina Superior Izquierda
            cv2.line(frame, (x1, y1), (x1 + l, y1), color_caja, t)
            cv2.line(frame, (x1, y1), (x1, y1 + l), color_caja, t)
            
            # Esquina Superior Derecha
            cv2.line(frame, (x2, y1), (x2 - l, y1), color_caja, t)
            cv2.line(frame, (x2, y1), (x2, y1 + l), color_caja, t)
            
            # Esquina Inferior Izquierda
            cv2.line(frame, (x1, y2), (x1 + l, y2), color_caja, t)
            cv2.line(frame, (x1, y2), (x1, y2 - l), color_caja, t)
            
            # Esquina Inferior Derecha
            cv2.line(frame, (x2, y2), (x2 - l, y2), color_caja, t)
            cv2.line(frame, (x2, y2), (x2, y2 - l), color_caja, t)

        return frame