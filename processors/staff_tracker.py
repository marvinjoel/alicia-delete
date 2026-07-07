import time
import cv2
import math
from processors.base_processor import BaseProcessor


class StaffTrackerProcessor(BaseProcessor):
    """
    Rastrea al personal en el piso de producción.
    Punto 4:
    - Calcula la distancia recorrida en píxeles (math.hypot).
    - Mide tiempos de ciclo en la estación de encajado.
    - Mantiene la lógica original de inactividad de operarios.
    """

    def __init__(self, camara_id, config):
        super().__init__(camara_id, config)
        self.historial_zonas = {}
        
        # --- NUEVAS VARIABLES: PUNTO 4 ---
        self.operarios = {}  # Memoria del tracker (ID -> datos)
        self.next_id = 1
        
        # Zona de encajado donde ocurre el ciclo (se lee del panel web)
        # Formato esperado: [x1, y1, x2, y2]
        self.zona_encajado = self.config_extra.get("zona_encajado", [100, 200, 400, 500])
        self.objetivo_ciclo = float(self.config_extra.get("objetivo_ciclo", 22.5))

    def procesar(self, frame, resultados):
        nombres = resultados[0].names
        cajas   = resultados[0].boxes
        ahora   = time.time()

        # 1. Extraer solo a las personas válidas
        personas_validas = []
        for box in cajas:
            if nombres[int(box.cls[0])] == "person" and float(box.conf[0]) > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                personas_validas.append((cx, cy, x1, y1, x2, y2))

        # Dibujar la Zona de Encajado en pantalla
        zx1, zy1, zx2, zy2 = self.zona_encajado
        cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (255, 144, 30), 2)
        self._texto(frame, "ZONA ENCAJADO", (zx1, zy1 - 10), (255, 144, 30), 0.5)

        cajas_asignadas = set()
        operarios_activos = set()

        # 2. Centroid Tracker: Actualizar posiciones y medir distancia (math.hypot)
        for op_id, data in self.operarios.items():
            mejor_idx = -1
            menor_dist = 100.0  # Tolerancia máxima de movimiento en px entre frames

            for i, (cx, cy, x1, y1, x2, y2) in enumerate(personas_validas):
                if i in cajas_asignadas:
                    continue
                distancia = math.hypot(cx - data["cx"], cy - data["cy"])
                if distancia < menor_dist:
                    menor_dist = distancia
                    mejor_idx = i

            if mejor_idx != -1:
                cajas_asignadas.add(mejor_idx)
                operarios_activos.add(op_id)
                
                nuevo_cx, nuevo_cy, bx1, by1, bx2, by2 = personas_validas[mejor_idx]
                
                # Calcular y sumar distancia real movida (filtrando micro-temblores < 2px)
                dist_movida = math.hypot(nuevo_cx - data["cx"], nuevo_cy - data["cy"])
                if dist_movida > 2.0:
                    self.operarios[op_id]["distancia_total"] += dist_movida
                
                self.operarios[op_id]["cx"] = nuevo_cx
                self.operarios[op_id]["cy"] = nuevo_cy
                self.operarios[op_id]["visto"] = ahora
                self.operarios[op_id]["box"] = (bx1, by1, bx2, by2)

        # 3. Registrar nuevos operarios que entran a cámara
        for i, (cx, cy, x1, y1, x2, y2) in enumerate(personas_validas):
            if i not in cajas_asignadas:
                nuevo_id = self.next_id
                self.next_id += 1
                self.operarios[nuevo_id] = {
                    "cx": cx, "cy": cy, "visto": ahora,
                    "box": (x1, y1, x2, y2),
                    "distancia_total": 0,
                    "estado_encajado": "fuera",
                    "inicio_ciclo": 0
                }
                operarios_activos.add(nuevo_id)

        # Limpiar memoria de los que salieron del encuadre por más de 10s
        self.operarios = {k: v for k, v in self.operarios.items() if (ahora - v["visto"]) < 10.0}

        # 4. Evaluación Lógica de Ciclos e Inactividad
        for op_id in operarios_activos:
            data = self.operarios[op_id]
            cx, cy = data["cx"], data["cy"]
            bx1, by1, bx2, by2 = data["box"]

            # --- A. LÓGICA DE INACTIVIDAD (Original) ---
            zona_grid = f"{bx1 // 150}_{by1 // 150}"
            if zona_grid not in self.historial_zonas:
                self.historial_zonas[zona_grid] = {"ultima_vez": ahora, "conteo": 1}
            else:
                self.historial_zonas[zona_grid]["ultima_vez"] = ahora
                self.historial_zonas[zona_grid]["conteo"] += 1

            # --- B. LÓGICA DE CICLO DE ENCAJADO (Nueva) ---
            en_zona = (zx1 < cx < zx2) and (zy1 < cy < zy2)

            if en_zona and data["estado_encajado"] == "fuera":
                # Entró a la zona: Inicia el cronómetro del ciclo
                data["estado_encajado"] = "dentro"
                data["inicio_ciclo"] = ahora

            elif not en_zona and data["estado_encajado"] == "dentro":
                # Salió de la zona: Termina el ciclo
                tiempo_ciclo = ahora - data["inicio_ciclo"]
                data["estado_encajado"] = "fuera"
                
                # Filtro: un ciclo real toma al menos 5 segundos. Evita falsos positivos de gente pasando.
                if tiempo_ciclo > 5.0:
                    exito = "completado" if tiempo_ciclo <= self.objetivo_ciclo else "retrasado"
                    
                    # EVENTO EXACTO QUE PIDE EL FRONTEND
                    datos_ciclo = {
                        "tipo_ciclo": "encajado",
                        "operario_id": op_id,
                        "tiempo_ciclo": round(tiempo_ciclo, 2),
                        "distancia_px": int(data["distancia_total"]),
                        "estado": exito
                    }
                    self._emitir_evento("ciclo", int(tiempo_ciclo), datos_ciclo)
                    print(f"[{self.modulo}] Cam={self.camara_id} | Ciclo de Encajado: {tiempo_ciclo:.1f}s ({exito})")

            # --- UI: Etiquetas sobre el operario ---
            dist_visual = int(data["distancia_total"] / 10) # Pseudo-escala para no mostrar miles de px
            if data["estado_encajado"] == "dentro":
                t_actual = ahora - data["inicio_ciclo"]
                lbl = f"OP-{op_id} | Encajando: {t_actual:.1f}s"
                c_box = (0, 255, 255) # Amarillo trabajando
            else:
                lbl = f"OP-{op_id} | Dist: {dist_visual}m"
                c_box = (200, 200, 200)

            cv2.rectangle(frame, (bx1, by1), (bx2, by2), c_box, 2)
            self._texto(frame, lbl, (bx1, max(by1 - 10, 20)), c_box, 0.5)

        # Evaluar y emitir alertas de inactividad de la zona_grid
        for zona_id, info in self.historial_zonas.items():
            if ahora - info["ultima_vez"] < 2 and info["conteo"] > 10:
                inactivo_seg = info["conteo"] * 0.1
                if inactivo_seg >= self.umbral_seg:
                    self._emitir_evento("operario_inactivo", int(inactivo_seg), {"zona": zona_id})

        # Limpiar historial_zonas viejo
        self.historial_zonas = {z: v for z, v in self.historial_zonas.items() if ahora - v["ultima_vez"] <= 30}

        # Panel General
        self._panel(frame, f"Operarios Activos: {len(operarios_activos)}", x=10, y=90, color_fondo=(60, 60, 150))

        return frame