import cv2
import numpy as np
from processors.base_processor import BaseProcessor


class WipTrackerProcessor(BaseProcessor):
    """
    Detecta acumulación de inventario en proceso (WIP - Work In Process).
    Cuenta objetos dentro de una zona específica y genera alertas si 
    superan el límite permitido (Cuello de botella / Sobreproducción).
    """
    def __init__(self, camara_id, config):
        super().__init__(camara_id, config)
        
        # Parámetros desde la base de datos o por defecto
        # ¿Qué clase de YOLO vamos a contar? (Ej: 0=persona, 39=botella. Si tienes un modelo entrenado para "cajas", pon el ID de la caja)
        self.clase_wip = int(self.config_extra.get("clase_wip", 39)) 
        
        # Límite de objetos antes de considerar que hay acumulación/cuello de botella
        self.limite_acumulacion = int(self.config_extra.get("limite_acumulacion", 5))
        
        # Zona donde se acumula el material (Polígono)
        zona_str = self.config_extra.get("zona_wip", "100,100;400,100;400,400;100,400")
        self.zona_poligono = self._parse_polygon(zona_str)

    def _parse_polygon(self, polygon_str):
        try:
            puntos = []
            for pt in polygon_str.split(';'):
                x, y = pt.split(',')
                puntos.append([int(x), int(y)])
            return np.array(puntos, np.int32)
        except:
            return np.array([[100, 100], [400, 100], [400, 400], [100, 400]], np.int32)

    def procesar(self, frame, resultados):
        if not resultados or not resultados[0].boxes:
            return frame

        conteo_wip = 0
        boxes = resultados[0].boxes

        # Dibujar la zona de acumulación (WIP)
        cv2.polylines(frame, [self.zona_poligono], isClosed=True, color=(255, 255, 0), thickness=2)
        self._texto(frame, f"Zona WIP (Max: {self.limite_acumulacion})", tuple(self.zona_poligono[0]), (255, 255, 0))

        for box in boxes:
            clase_id = int(box.cls[0])
            
            # Solo nos importan los objetos configurados como "Inventario" (ej. cajas, botellas)
            if clase_id == self.clase_wip:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Calcular el centro del objeto
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Verificar si el objeto está dentro de la zona WIP
                adentro = cv2.pointPolygonTest(self.zona_poligono, (cx, cy), False) >= 0
                
                if adentro:
                    conteo_wip += 1
                    # Dibujar caja del objeto contado
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        # Evaluar regla de acumulación (Muda por exceso de inventario)
        if conteo_wip > self.limite_acumulacion:
            alerta = f"ALERTA: Cuello de botella. WIP = {conteo_wip}"
            color_alerta = (0, 0, 255) # Rojo
            
            # Disparar evento a la BD
            self._emitir_evento("acumulacion_wip", conteo_wip, {
                "conteo_actual": conteo_wip,
                "limite": self.limite_acumulacion,
                "estado": "cuello_de_botella"
            })
        else:
            alerta = f"WIP OK: {conteo_wip}/{self.limite_acumulacion}"
            color_alerta = (0, 255, 0) # Verde

        # Mostrar el conteo general en pantalla
        self._texto(frame, alerta, (50, 50), color_alerta, 1.0)

        return frame