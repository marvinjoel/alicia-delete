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
        
        self.zonas_mesas = config.get("mesas", {})

    def procesar(self, frame, resultados):
       
        if not self.zonas_mesas:
            cv2.putText(frame, "Esperando configuracion de mesas...", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame

        nombres = resultados[0].names
        cajas   = resultados[0].boxes
        ahora   = time.time()

        personas_en_mesas = set()
        comida_en_mesas = set()

        # 1. Asignar detecciones a las mesas dibujadas
        for box in cajas:
            if float(box.conf[0]) < 0.4: 
                continue
                
            clase_nom = nombres[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Punto central del objeto
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Revisar si el centro del objeto cae dentro de alguna mesa
            mesa_detectada = None
            for nombre_mesa, coords in self.zonas_mesas.items():
                if (coords["x1"] <= cx <= coords["x2"]) and (coords["y1"] <= cy <= coords["y2"]):
                    mesa_detectada = nombre_mesa
                    break
            
            if mesa_detectada:
                if clase_nom == "person":
                    personas_en_mesas.add(mesa_detectada)
                elif clase_nom in CLASES_COMIDA:
                    comida_en_mesas.add(mesa_detectada)

        # 2. Lógica de cronómetros por cada mesa registrada
        for nombre_mesa, coords in self.zonas_mesas.items():
            # Extraer coordenadas para dibujar gráficos
            mx1, my1, mx2, my2 = coords["x1"], coords["y1"], coords["x2"], coords["y2"]
            
            # Dibujar el rectángulo de la mesa
            cv2.rectangle(frame, (mx1, my1), (mx2, my2), (255, 0, 0), 2)
            cv2.putText(frame, nombre_mesa, (mx1, my1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Si hay una persona en esa mesa
            if nombre_mesa in personas_en_mesas:
                if nombre_mesa not in self.mesas_estado:
                    # Cliente nuevo: Arranca el tiempo
                    self.mesas_estado[nombre_mesa] = {"estado": "esperando", "inicio": ahora, "ultimo_visto": ahora}
                else:
                    self.mesas_estado[nombre_mesa]["ultimo_visto"] = ahora
                    
                    if self.mesas_estado[nombre_mesa]["estado"] == "esperando":
                        tiempo_espera = int(ahora - self.mesas_estado[nombre_mesa]["inicio"])
                        
                        # Dibujar contador en amarillo
                        txt_tiempo = f"Espera: {tiempo_espera}s"
                        cv2.putText(frame, txt_tiempo, (mx1, my1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

                        # Si detectamos comida, cambiamos el estado a servido y disparamos el guardado
                        if nombre_mesa in comida_en_mesas:
                            self.mesas_estado[nombre_mesa]["estado"] = "servido"
                            
                            # ESTO ES CLAVE: Disparamos el evento para guardar en la nueva tabla
                            datos_evento = {
                                "fk_camara": self.camara_id,
                                "nombre_mesa": nombre_mesa,
                                "tiempo_espera_segundos": tiempo_espera
                            }
                            self._emitir_evento("comida_servida", tiempo_espera, datos_evento)
                            print(f"[Cam {self.camara_id}] ¡Mesa {nombre_mesa} servida en {tiempo_espera}s!")
                    
                    elif self.mesas_estado[nombre_mesa]["estado"] == "servido":
                        # Dibujar en verde si ya le llevaron comida
                        cv2.putText(frame, "SERVIDO", (mx1, my1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 3. Limpieza: Si una mesa pasa 60 segundos vacía, reseteamos el sistema para clientes nuevos
        inactivas = [m for m, data in self.mesas_estado.items() if (ahora - data["ultimo_visto"]) > 60]
        for m in inactivas:
            del self.mesas_estado[m]

        return frame