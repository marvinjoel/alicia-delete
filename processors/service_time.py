import time
import cv2
from processors.base_processor import BaseProcessor

CLASES_COMIDA = {
    "cup", "bowl", "bottle", "wine glass", "plate", 
    "fork", "knife", "spoon", "pizza", "hot dog", 
    "sandwich", "donut", "cake"
}

class ServiceTimeProcessor(BaseProcessor):
    """
    Mide el tiempo de servicio cruzando las detecciones con las Zonas (Mesas) 
    dibujadas por el frontend y guardadas en el config_json de la BD.
    """
    def __init__(self, camara_id, config) -> None:
        super().__init__(camara_id, config)
        self.mesas_estado = {}
        
        self.zonas_mesas = self.config_extra.get("zonas", {})

    def procesar(self, frame, resultados):
        
        if not self.zonas_mesas:
            cv2.putText(frame, "Esperando configuracion de mesas...", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame

        nombres = resultados[0].names
        cajas   = resultados[0].boxes
        ahora   = time.time()

        # 1. CONVERSIÓN DE ESCALA (Magia del Backend)
        # Extraemos el alto y ancho real del video en este momento
        alto_frame, ancho_frame = frame.shape[:2]
        
        mesas_pixeles = {}
        for nombre_mesa, coords in self.zonas_mesas.items():
            # Asumimos que el front nos manda porcentajes de 0 a 100
            # Ej: Si x1 = 50 (la mitad), y el video mide 1920, 50/100 * 1920 = 960px
            mx1 = int((coords["x1"] / 100.0) * ancho_frame)
            my1 = int((coords["y1"] / 100.0) * alto_frame)
            mx2 = int((coords["x2"] / 100.0) * ancho_frame)
            my2 = int((coords["y2"] / 100.0) * alto_frame)
            
            mesas_pixeles[nombre_mesa] = {"x1": mx1, "y1": my1, "x2": mx2, "y2": my2}

        personas_en_mesas = set()
        comida_en_mesas = set()

        # 2. Asignar detecciones a las mesas dibujadas (usando colisión real de cajas)
        for box in cajas:
            confianza = float(box.conf[0])
            clase_nom = nombres[int(box.cls[0])]
            
            if clase_nom == "person" and confianza < 0.4: 
                continue
            if clase_nom in CLASES_COMIDA and confianza < 0.2:
                continue
            if clase_nom != "person" and clase_nom not in CLASES_COMIDA:
                continue
                
            # Coordenadas del objeto (persona o comida) detectado por la IA
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            mesa_detectada = None
            for nombre_mesa, mp in mesas_pixeles.items():
                mx1, my1, mx2, my2 = mp["x1"], mp["y1"], mp["x2"], mp["y2"]
                
                # Le damos 40 píxeles de "aura" o margen invisible a la mesa. 
                # Porque la gente se sienta AFUERA de la mesa y los platos se ponen en los bordes.
                margen = 40 
                
                # LÓGICA DE COLISIÓN DE CAJAS: ¿El objeto toca el área de la mesa?
                if (x1 < (mx2 + margen) and x2 > (mx1 - margen) and 
                    y1 < (my2 + margen) and y2 > (my1 - margen)):
                    mesa_detectada = nombre_mesa
                    break
            
            if mesa_detectada:
                if clase_nom == "person":
                    personas_en_mesas.add(mesa_detectada)
                elif clase_nom in CLASES_COMIDA:
                    comida_en_mesas.add(mesa_detectada)

        # 3. Lógica de cronómetros por cada mesa registrada
        for nombre_mesa, mp in mesas_pixeles.items():
            mx1, my1, mx2, my2 = mp["x1"], mp["y1"], mp["x2"], mp["y2"]
            
            # Dibujar el rectángulo de la mesa en su lugar exacto
            cv2.rectangle(frame, (mx1, my1), (mx2, my2), (255, 0, 0), 2)
            
            # Función rápida para dibujar texto con borde negro (para que resalte)
            def texto_borde(txt, px, py, color):
                cv2.putText(frame, txt, (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4) # Sombra negra gruesa
                cv2.putText(frame, txt, (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)     # Texto de color encima

            texto_borde(nombre_mesa, mx1, my1 - 10, (255, 255, 255)) # Blanco

            if nombre_mesa in personas_en_mesas:
                if nombre_mesa not in self.mesas_estado:
                    self.mesas_estado[nombre_mesa] = {"estado": "esperando", "inicio": ahora, "ultimo_visto": ahora}
                else:
                    self.mesas_estado[nombre_mesa]["ultimo_visto"] = ahora
                    
                    if self.mesas_estado[nombre_mesa]["estado"] == "esperando":
                        tiempo_espera = int(ahora - self.mesas_estado[nombre_mesa]["inicio"])
                        
                        texto_borde(f"Espera: {tiempo_espera}s", mx1, my1 - 35, (0, 200, 255))

                        if nombre_mesa in comida_en_mesas:
                            self.mesas_estado[nombre_mesa]["estado"] = "servido"
                            
                            datos_evento = {
                                "fk_camara": self.camara_id,
                                "nombre_mesa": nombre_mesa,
                                "tiempo_espera_segundos": tiempo_espera
                            }
                            self._emitir_evento("comida_servida", tiempo_espera, datos_evento)
                            print(f"[Cam {self.camara_id}] ¡Mesa {nombre_mesa} servida en {tiempo_espera}s!")
                    
                    elif self.mesas_estado[nombre_mesa]["estado"] == "servido":
                        texto_borde("SERVIDO", mx1, my1 - 35, (0, 255, 0))

        # 4. Limpieza de mesas vacías (>60 seg inactividad)
        inactivas = [m for m, data in self.mesas_estado.items() if (ahora - data["ultimo_visto"]) > 60]
        for m in inactivas:
            del self.mesas_estado[m]

        return frame