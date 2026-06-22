import time
import cv2
import math
from processors.base_processor import BaseProcessor

class EfficiencyTrackerProcessor(BaseProcessor):
    """
    Rastrea la eficiencia de operarios IGNORANDO los IDs de YOLO.
    Usa un Centroid Tracker propio, indestructible ante cortes de red.
    AHORA CON MEDICIÓN DE DISTANCIA RECORRIDA (MVP-Fase 1).
    """
    def __init__(self, camara_id, config):
        super().__init__(camara_id, config)
        self.zonas = self.config_extra.get("zonas", {})
        
        self.operarios = {}
        self.next_id = 1

    def _actualizar_maquina_estados(self, op_data, en_a, en_b, ahora, op_id):
        estado = op_data["estado"]
        ciclo = op_data["ciclo"]
        etiqueta = ""

        if en_a:
            # 1. CIERRE DEL CICLO NVA (Retorno a la mesa)
            if estado == "viajando_a_a":
                t_ida = ciclo["fin_ida"] - ciclo["inicio_ida"]
                t_busqueda = ciclo["inicio_retorno"] - ciclo["fin_ida"]
                t_retorno = ahora - ciclo["inicio_retorno"]

                if t_ida > 0 and t_busqueda >= 0:
                    t_retorno = max(1, int(t_retorno))
                    tiempo_total = int(t_ida + t_busqueda + t_retorno)
                    
                    dist_ida = int(ciclo.get("dist_ida", 0))
                    dist_retorno = int(ciclo.get("dist_retorno", 0))
                    dist_total = dist_ida + dist_retorno

                    # --- CLASIFICACIÓN LEAN: DESPERDICIO (NVA) ---
                    datos_ciclo = {
                        "operario_id": op_id,
                        "clasificacion": "No Valor Agregado (NVA)",
                        "nva_traslado_seg": int(t_ida) + t_retorno,
                        "nva_espera_seg": int(t_busqueda),
                        "tiempo_total": tiempo_total,
                        "distancia_ida_px": dist_ida,
                        "distancia_retorno_px": dist_retorno,
                        "distancia_total_px": dist_total
                    }
                    self._emitir_evento("ciclo_nva_completado", tiempo_total, datos_ciclo)
                    print(f"[Cam {self.camara_id}] Desperdicio NVA registrado: {datos_ciclo}")

                # Reseteamos ciclo de viaje y arranca cronómetro de VALOR AGREGADO
                op_data["ciclo"] = {
                    "inicio_ida": 0, "fin_ida": 0, "inicio_retorno": 0, 
                    "dist_ida": 0, "dist_retorno": 0,
                    "inicio_mesa": ahora # <-- Inicia el tiempo productivo
                }

            # Si es un operario que recién aparece directamente en la mesa
            elif "inicio_mesa" not in ciclo or ciclo["inicio_mesa"] == 0:
                ciclo["inicio_mesa"] = ahora

            op_data["estado"] = "en_mesa"
            
            # Mostramos en pantalla el tiempo productivo en vivo
            t_va = int(ahora - ciclo.get("inicio_mesa", ahora))
            etiqueta = f" (VA Productivo: {t_va}s)"

        elif en_b:
            # 2. EN LAS HERRAMIENTAS (NVA - Espera / Búsqueda)
            if estado == "viajando_a_b":
                ciclo["fin_ida"] = ahora
                op_data["estado"] = "en_herramientas"
            
            if ciclo["fin_ida"] > 0:
                t_busq = int(ahora - ciclo["fin_ida"])
                etiqueta = f" (NVA Búsqueda: {t_busq}s)"
            else:
                op_data["estado"] = "en_herramientas"
                etiqueta = " (NVA Búsqueda)"

        else:
            # 3. EN TRANSICIÓN (Pasillos = NVA Traslado)
            if estado == "en_mesa":
                # --- CLASIFICACIÓN LEAN: VALOR AGREGADO (VA) ---
                # El operario acaba de levantarse de la mesa. Cerramos el ciclo de trabajo.
                t_mesa = ahora - ciclo.get("inicio_mesa", ahora)
                if t_mesa > 2: # Filtro para evitar milisegundos falsos
                    datos_va = {
                        "operario_id": op_id,
                        "clasificacion": "Valor Agregado (VA)",
                        "tiempo_mesa_seg": int(t_mesa)
                    }
                    self._emitir_evento("actividad_va_completada", int(t_mesa), datos_va)
                    print(f"[Cam {self.camara_id}] Trabajo VA registrado: {datos_va}")

                # Iniciamos viaje
                op_data["estado"] = "viajando_a_b"
                ciclo["inicio_ida"] = ahora
                ciclo["fin_ida"] = 0
                ciclo["inicio_retorno"] = 0
                ciclo["dist_ida"] = 0
                ciclo["dist_retorno"] = 0
                ciclo["inicio_mesa"] = 0 # Reseteamos la mesa

            elif estado == "en_herramientas":
                op_data["estado"] = "viajando_a_a"
                ciclo["inicio_retorno"] = ahora

            if op_data["estado"] == "viajando_a_b":
                t_ida = int(ahora - ciclo["inicio_ida"])
                dist = int(ciclo.get("dist_ida", 0))
                etiqueta = f" (NVA Ida: {t_ida}s | {dist}px)"
            elif op_data["estado"] == "viajando_a_a":
                t_retorno = int(ahora - ciclo["inicio_retorno"])
                dist = int(ciclo.get("dist_retorno", 0))
                etiqueta = f" (NVA Retorno: {t_retorno}s | {dist}px)"
            else:
                etiqueta = " (Caminando)"

        return etiqueta

    def procesar(self, frame, resultados):
        if not self.zonas or len(self.zonas) < 2:
            return frame

        nombres = resultados[0].names
        cajas = resultados[0].boxes
        alto, ancho = frame.shape[:2]
        ahora = time.time()

        nombres_zonas = list(self.zonas.keys())
        nom_a, nom_b = nombres_zonas[0], nombres_zonas[1]
        
        def px(porcentaje, maximo): return int((porcentaje / 100.0) * maximo)
        
        caja_a = {"x1": px(self.zonas[nom_a]["x1"], ancho), "y1": px(self.zonas[nom_a]["y1"], alto), "x2": px(self.zonas[nom_a]["x2"], ancho), "y2": px(self.zonas[nom_a]["y2"], alto)}
        caja_b = {"x1": px(self.zonas[nom_b]["x1"], ancho), "y1": px(self.zonas[nom_b]["y1"], alto), "x2": px(self.zonas[nom_b]["x2"], ancho), "y2": px(self.zonas[nom_b]["y2"], alto)}

        def dibujar_etiqueta(img, texto, x, y, color_fondo, color_texto=(255, 255, 255)):
            (w_txt, h_txt), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x, y - h_txt - 8), (x + w_txt + 10, y + 2), color_fondo, -1)
            cv2.putText(img, texto, (x + 5, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_texto, 1, cv2.LINE_AA)

        cv2.rectangle(frame, (caja_a["x1"], caja_a["y1"]), (caja_a["x2"], caja_a["y2"]), (255, 144, 30), 2)
        dibujar_etiqueta(frame, nom_a.upper(), caja_a["x1"], caja_a["y1"], (255, 144, 30), (0,0,0))
        
        cv2.rectangle(frame, (caja_b["x1"], caja_b["y1"]), (caja_b["x2"], caja_b["y2"]), (30, 144, 255), 2)
        dibujar_etiqueta(frame, nom_b.upper(), caja_b["x1"], caja_b["y1"], (30, 144, 255), (0,0,0))

        # 1. Extraer a las personas puras (IGNORAMOS box.id)
        cajas_validas = []
        for box in cajas:
            clase_nom = nombres[int(box.cls[0])]
            confianza = float(box.conf[0])
            
            if clase_nom == "person" and confianza > 0.4:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                cx = (bx1 + bx2) // 2
                cy = by1 + int((by2 - by1) * 0.8)
                cajas_validas.append((cx, cy, bx1, by1, bx2, by2))

        # 2. Emparejar las cajas visuales con nuestra memoria interna
        cajas_asignadas = set()
        operarios_asignados = set()

        operarios_ordenados = sorted(self.operarios.items(), key=lambda x: x[1]["visto"], reverse=True)

        for op_id, data in operarios_ordenados:
            mejor_caja_idx = -1
            
            # Ajuste: 90 píxeles. Suficiente para absorber el "temblor" cuando dos personas se cruzan,
            # pero no tan grande como para robar IDs de otras mesas.
            menor_distancia = 90.0 

            for i, (cx, cy, bx1, by1, bx2, by2) in enumerate(cajas_validas):
                if i in cajas_asignadas:
                    continue
                
                distancia = math.hypot(cx - data["cx"], cy - data["cy"])
                if distancia < menor_distancia:
                    menor_distancia = distancia
                    mejor_caja_idx = i

            if mejor_caja_idx != -1:
                cajas_asignadas.add(mejor_caja_idx)
                operarios_asignados.add(op_id)
                
                # --- AQUÍ SUCEDE LA MAGIA DE LA DISTANCIA ---
                nuevo_cx = cajas_validas[mejor_caja_idx][0]
                nuevo_cy = cajas_validas[mejor_caja_idx][1]
                
                distancia_pixel = math.hypot(nuevo_cx - data["cx"], nuevo_cy - data["cy"])
                
                # Filtro anti-ruido
                if distancia_pixel > 2.0:
                    estado_actual = data["estado"]
                    if estado_actual == "viajando_a_b":
                        data["ciclo"]["dist_ida"] = data["ciclo"].get("dist_ida", 0) + distancia_pixel
                    elif estado_actual == "viajando_a_a":
                        data["ciclo"]["dist_retorno"] = data["ciclo"].get("dist_retorno", 0) + distancia_pixel
                
                # Actualizamos su ubicación
                self.operarios[op_id]["cx"] = nuevo_cx
                self.operarios[op_id]["cy"] = nuevo_cy
                self.operarios[op_id]["visto"] = ahora
                self.operarios[op_id]["box"] = cajas_validas[mejor_caja_idx]

        # 3. Limpieza de memoria (Efecto Persistencia)
        self.operarios = {k: v for k, v in self.operarios.items() if (ahora - v["visto"]) < 15.0}

        # 4. Nacimiento de NUEVOS operarios
        for i, (cx, cy, bx1, by1, bx2, by2) in enumerate(cajas_validas):
            if i not in cajas_asignadas:
                nuevo_id = self.next_id
                self.next_id += 1
                
                self.operarios[nuevo_id] = {
                    "cx": cx, "cy": cy, "visto": ahora,
                    "estado": "neutro",
                    "ciclo": {
                        "inicio_ida": 0, "fin_ida": 0, "inicio_retorno": 0,
                        "dist_ida": 0, "dist_retorno": 0, 
                        "inicio_mesa": 0
                    },
                    "box": (cx, cy, bx1, by1, bx2, by2)
                }
                operarios_asignados.add(nuevo_id)

        # =========================================================
        # EJECUCIÓN DE CRONÓMETROS Y UI
        # =========================================================
        for op_id in operarios_asignados:
            if op_id in self.operarios and "box" in self.operarios[op_id]:
                if self.operarios[op_id]["visto"] == ahora:
                    cx, cy, bx1, by1, bx2, by2 = self.operarios[op_id]["box"]

                    en_a = caja_a["x1"] < cx < caja_a["x2"] and caja_a["y1"] < cy < caja_a["y2"]
                    en_b = caja_b["x1"] < cx < caja_b["x2"] and caja_b["y1"] < cy < caja_b["y2"]
                    
                    etiqueta_estado = self._actualizar_maquina_estados(self.operarios[op_id], en_a, en_b, ahora, op_id)

                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (50, 205, 50), 1)
                    dibujar_etiqueta(frame, f"OP-{op_id}{etiqueta_estado}", bx1, by1, (40, 40, 40))

        return frame