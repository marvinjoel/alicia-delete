import time
import cv2
from processors.base_processor import BaseProcessor

CLASES_COMIDA = {
    "bowl", "pizza", "hot dog", "sandwich", 
    "donut", "cake", "fork", "knife", "spoon"
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

        alto_frame, ancho_frame = frame.shape[:2]
        mesas_pixeles = {}
        for nombre_mesa, coords in self.zonas_mesas.items():
            mx1 = int((coords["x1"] / 100.0) * ancho_frame)
            my1 = int((coords["y1"] / 100.0) * alto_frame)
            mx2 = int((coords["x2"] / 100.0) * ancho_frame)
            my2 = int((coords["y2"] / 100.0) * alto_frame)
            mesas_pixeles[nombre_mesa] = {"x1": mx1, "y1": my1, "x2": mx2, "y2": my2}

        personas_en_mesas = set()
        comida_en_mesas = set()

        # 2. Asignar detecciones a las mesas dibujadas
        for box in cajas:
            confianza = float(box.conf[0])
            clase_nom = nombres[int(box.cls[0])]
            
            if clase_nom == "person" and confianza < 0.4: 
                continue
            # EXIGENCIA AL 8% (0.08): Forzamos la vista al máximo para atrapar manchas del fondo
            if clase_nom in CLASES_COMIDA and confianza < 0.08: 
                continue
            if clase_nom != "person" and clase_nom not in CLASES_COMIDA:
                continue
                
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            mesa_detectada = None
            
            if clase_nom in CLASES_COMIDA:
                # RAYOS X: Dibujamos en ROSADO lo que la IA considera comida/cubiertos
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(frame, f"{clase_nom} {int(confianza*100)}%", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

                # REGLA PARA COMIDA: Margen pequeño (15px)
                for nombre_mesa, mp in mesas_pixeles.items():
                    if (mp["x1"] - 15 <= cx <= mp["x2"] + 15) and (mp["y1"] - 15 <= cy <= mp["y2"] + 15):
                        mesa_detectada = nombre_mesa
                        break
            else:
                # REGLA PARA PERSONAS
                mayor_area = 0
                for nombre_mesa, mp in mesas_pixeles.items():
                    margen = 40
                    ix1 = max(x1, mp["x1"] - margen)
                    iy1 = max(y1, mp["y1"] - margen)
                    ix2 = min(x2, mp["x2"] + margen)
                    iy2 = min(y2, mp["y2"] + margen)
                    
                    if ix1 < ix2 and iy1 < iy2:
                        area_choque = (ix2 - ix1) * (iy2 - iy1)
                        if area_choque > mayor_area:
                            mayor_area = area_choque
                            mesa_detectada = nombre_mesa
            
            if mesa_detectada:
                if clase_nom == "person":
                    personas_en_mesas.add(mesa_detectada)
                elif clase_nom in CLASES_COMIDA:
                    comida_en_mesas.add(mesa_detectada)

        # 3. Lógica de cronómetros
        for nombre_mesa, mp in mesas_pixeles.items():
            mx1, my1, mx2, my2 = mp["x1"], mp["y1"], mp["x2"], mp["y2"]
            cv2.rectangle(frame, (mx1, my1), (mx2, my2), (255, 0, 0), 2)
            
            def texto_borde(txt, px, py, color):
                cv2.putText(frame, txt, (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                cv2.putText(frame, txt, (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            texto_borde(nombre_mesa, mx1, my1 - 10, (255, 255, 255))

            if nombre_mesa in personas_en_mesas:
                if nombre_mesa not in self.mesas_estado:
                    self.mesas_estado[nombre_mesa] = {
                        "estado": "esperando", 
                        "inicio": ahora, 
                        "ultimo_visto": ahora,
                        "ultimo_visto_comida": 0
                    }
                else:
                    self.mesas_estado[nombre_mesa]["ultimo_visto"] = ahora
                    
                    if nombre_mesa in comida_en_mesas:
                        self.mesas_estado[nombre_mesa]["ultimo_visto_comida"] = ahora

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
                        # 120 SEGUNDOS DE MEMORIA
                        if ahora - self.mesas_estado[nombre_mesa]["ultimo_visto_comida"] > 120:
                            self.mesas_estado[nombre_mesa]["estado"] = "esperando"
                        else:
                            texto_borde("SERVIDO", mx1, my1 - 35, (0, 255, 0))

        # 4. Limpieza de mesas vacías
        inactivas = [m for m, data in self.mesas_estado.items() if (ahora - data["ultimo_visto"]) > 60]
        for m in inactivas:
            del self.mesas_estado[m]

        return frame