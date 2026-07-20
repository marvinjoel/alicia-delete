import os
import time
import math
import cv2
from processors.base_processor import BaseProcessor

# Objetos que representan un producto que un cliente/empleado puede llevar.
CLASES_PRODUCTO = {"cup", "bottle", "wine glass", "bowl"}
CONFIANZA_MINIMA = 0.5

PALABRAS_ZONA_CAJA = ("caja", "cobro", "registro")
PALABRAS_ZONA_RESTRINGIDA = ("restringid", "bodega", "trasera", "backoffice", "salida")


class BehaviorAnomalyTrackerProcessor(BaseProcessor):
    """
    Señales heuristicas de comportamiento A REVISAR — no son una acusacion
    automatica de hurto, son candidatos para que una persona revise el clip:

      - cobro_incompleto: alguien pasa por zona_caja con un producto pero se
        retira antes del tiempo minimo de un cobro real (umbral_cobro_seg).
      - objeto_sin_registrar: un producto que estuvo cerca de una persona
        desaparece sin haber pasado el tiempo minimo dentro de zona_caja.
      - desplazamiento_sospechoso: solo si se configura una zona_restringida
        (por nombre, ver PALABRAS_ZONA_RESTRINGIDA) — alguien que estuvo en
        zona_caja hace poco entra a esa zona. Aproximado por posicion, no hay
        identidad persistente de la persona (no se cruza con reconocimiento
        facial ni con otros procesadores).
      - billete_fuera_de_caja: siempre activo si existe models/yolo11_dinero.pt
        (igual que face_blur o ergonomics_tracker, sin toggle en config_json).
        Carga un modelo YOLO aparte (clase "billete") y corre una segunda
        pasada de inferencia independiente sobre el mismo frame. Si el archivo
        no existe, se desactiva solo esta señal, sin afectar al resto.
    """

    def __init__(self, camara_id, config):
        super().__init__(camara_id, config)
        self.zonas = self.config_extra.get("zonas", {})
        self.umbral_cobro_seg = float(self.config_extra.get("umbral_cobro_seg", 3.0))
        self.gracia_seg = float(self.config_extra.get("gracia_seg", 3.0))
        self.ventana_desplazamiento_seg = float(self.config_extra.get("ventana_desplazamiento_seg", 10.0))
        self.confianza_dinero = float(self.config_extra.get("confianza_dinero", 0.6))

        # persona_id -> {"cx","cy","x1","y1","x2","y2","inicio","visto","cerca_producto"}
        self.personas_en_caja = {}
        self.next_id_persona = 1
        # objeto_id -> {"cx","cy","x1","y1","x2","y2","inicio","visto","tiempo_en_caja","alguna_vez_con_persona"}
        self.objetos_en_caja = {}
        self.next_id_objeto = 1
        # Posiciones recientes de gente que acaba de salir de zona_caja (para desplazamiento_sospechoso)
        self.recientes_de_caja = []

        self.modelo_dinero = None
        self._cargar_modelo_dinero()

    def _cargar_modelo_dinero(self):
        ruta = "models/yolo11_dinero.pt"
        if not os.path.exists(ruta):
            print(f"[{self.modulo}] Cam={self.camara_id} | 🔴 No se encontro {ruta}, senal de dinero desactivada")
            return
        from ultralytics import YOLO
        self.modelo_dinero = YOLO(ruta)
        try:
            import torch
            dev_cfg = os.getenv("IA_DEVICE", "auto").lower()
            dev = "cuda" if dev_cfg == "cuda" or (dev_cfg == "auto" and torch.cuda.is_available()) else "cpu"
            self.modelo_dinero.to(dev)
            print(f"[{self.modulo}] Cam={self.camara_id} | modelo dinero cargado en {dev}")
        except Exception as e:
            print(f"[{self.modulo}] Cam={self.camara_id} | aviso device modelo dinero: {e}")

    def procesar(self, frame, resultados):
        if not self.zonas:
            if self._puede_loguear():
                print(f"[{self.modulo}] Cam={self.camara_id} | sin zona_caja configurada, omitiendo")
            return frame

        alto, ancho = frame.shape[:2]
        nombre_caja, nombre_restringida = self._identificar_zonas()
        if not nombre_caja:
            if self._puede_loguear():
                print(f"[{self.modulo}] Cam={self.camara_id} | no se pudo identificar zona_caja, omitiendo")
            return frame

        zona_caja = self._pct_a_px(self.zonas[nombre_caja], ancho, alto)
        zona_restringida = self._pct_a_px(self.zonas[nombre_restringida], ancho, alto) if nombre_restringida else None

        self._caja(frame, zona_caja["x1"], zona_caja["y1"], zona_caja["x2"], zona_caja["y2"], color=(0, 200, 255), grosor=1)
        self._etiqueta(frame, "CAJA", zona_caja["x1"], zona_caja["y1"], (0, 200, 255))
        if zona_restringida:
            self._caja(frame, zona_restringida["x1"], zona_restringida["y1"], zona_restringida["x2"], zona_restringida["y2"], color=(0, 0, 200), grosor=1)
            self._etiqueta(frame, nombre_restringida.upper(), zona_restringida["x1"], zona_restringida["y1"], (0, 0, 200))

        nombres = resultados[0].names
        cajas   = resultados[0].boxes
        ahora   = time.time()

        personas  = []
        productos = []
        for box in cajas:
            if float(box.conf[0]) < CONFIANZA_MINIMA:
                continue
            clase_nom = nombres[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if clase_nom == "person":
                personas.append((cx, cy, x1, y1, x2, y2))
            elif clase_nom in CLASES_PRODUCTO:
                productos.append((cx, cy, x1, y1, x2, y2))

        self._actualizar_personas_en_caja(frame, personas, productos, zona_caja, ahora)
        self._actualizar_objetos_en_caja(frame, productos, personas, zona_caja, ahora)

        if zona_restringida:
            self._revisar_desplazamiento(personas, zona_restringida, ahora)

        if self.modelo_dinero:
            self._revisar_billetes(frame, zona_caja, ahora)

        if self._puede_loguear():
            print(f"[{self.modulo}] Cam={self.camara_id} | personas_caja={len(self.personas_en_caja)} objetos_caja={len(self.objetos_en_caja)}")

        return frame

    # ── Regla 1 y base del tracking de personas en zona_caja ────────────────

    def _actualizar_personas_en_caja(self, frame, personas, productos, zona_caja, ahora):
        actuales = set()
        for cx, cy, x1, y1, x2, y2 in personas:
            if not self._dentro(cx, cy, zona_caja):
                continue
            p_id = self._obtener_o_crear(self.personas_en_caja, "next_id_persona", cx, cy, x1, y1, x2, y2, ahora)
            actuales.add(p_id)
            data = self.personas_en_caja[p_id]
            if not data.get("cerca_producto"):
                data["cerca_producto"] = any(
                    math.hypot(cx - ox, cy - oy) < 100 for ox, oy, *_ in productos
                )
            color = (0, 200, 0) if data["cerca_producto"] else (0, 200, 255)
            self._caja(frame, x1, y1, x2, y2, color, grosor=1)

        for p_id in list(self.personas_en_caja.keys()):
            if p_id in actuales:
                continue
            data = self.personas_en_caja[p_id]
            if ahora - data["visto"] < self.gracia_seg:
                continue  # posible parpadeo de deteccion

            del self.personas_en_caja[p_id]
            duracion = data["visto"] - data["inicio"]
            self.recientes_de_caja.append({"cx": data["cx"], "cy": data["cy"], "hora_salida": data["visto"]})

            if duracion < self.umbral_cobro_seg and data.get("cerca_producto"):
                self._emitir_evento(
                    "cobro_incompleto",
                    int(duracion),
                    {
                        "alerta": "revisar_manualmente",
                        "tiempo_en_caja": round(duracion, 1),
                        "umbral_cobro_seg": self.umbral_cobro_seg,
                    },
                    clave_evento=f"cobro_incompleto:{p_id}",
                )

        # Limpiar posiciones "recientes" fuera de la ventana de desplazamiento
        self.recientes_de_caja = [
            r for r in self.recientes_de_caja
            if ahora - r["hora_salida"] < self.ventana_desplazamiento_seg
        ]

    # ── Regla 2: producto que desaparece sin pasar por caja ─────────────────

    def _actualizar_objetos_en_caja(self, frame, productos, personas, zona_caja, ahora):
        actuales = set()
        for cx, cy, x1, y1, x2, y2 in productos:
            o_id = self._obtener_o_crear(self.objetos_en_caja, "next_id_objeto", cx, cy, x1, y1, x2, y2, ahora)
            actuales.add(o_id)
            data = self.objetos_en_caja[o_id]

            if self._dentro(cx, cy, zona_caja):
                data["tiempo_en_caja"] = data.get("tiempo_en_caja", 0.0) + (ahora - data["visto_anterior"])
            data["visto_anterior"] = ahora

            if not data.get("alguna_vez_con_persona"):
                data["alguna_vez_con_persona"] = any(
                    math.hypot(cx - px, cy - py) < 100 for px, py, *_ in personas
                )

        for o_id in list(self.objetos_en_caja.keys()):
            if o_id in actuales:
                continue
            data = self.objetos_en_caja[o_id]
            if ahora - data["visto"] < self.gracia_seg:
                continue

            del self.objetos_en_caja[o_id]
            if data.get("alguna_vez_con_persona") and data.get("tiempo_en_caja", 0.0) < self.umbral_cobro_seg:
                self._emitir_evento(
                    "objeto_sin_registrar",
                    int(ahora - data["inicio"]),
                    {
                        "alerta": "revisar_manualmente",
                        "tiempo_en_caja": round(data.get("tiempo_en_caja", 0.0), 1),
                    },
                    clave_evento=f"objeto_sin_registrar:{o_id}",
                )

    # ── Regla 3: desplazamiento hacia zona restringida tras estar en caja ───

    def _revisar_desplazamiento(self, personas, zona_restringida, ahora):
        for cx, cy, *_ in personas:
            if not self._dentro(cx, cy, zona_restringida):
                continue
            for reciente in self.recientes_de_caja:
                if math.hypot(cx - reciente["cx"], cy - reciente["cy"]) < 150:
                    self._emitir_evento(
                        "desplazamiento_sospechoso",
                        int(ahora - reciente["hora_salida"]),
                        {"alerta": "revisar_manualmente"},
                        clave_evento=f"desplazamiento:{int(reciente['cx'])}_{int(reciente['cy'])}",
                    )
                    break

    # ── Regla 4 (opcional): billete detectado fuera de zona_caja ────────────

    def _revisar_billetes(self, frame, zona_caja, ahora):
        res = self.modelo_dinero(frame, verbose=False, conf=self.confianza_dinero)[0]
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if self._dentro(cx, cy, zona_caja):
                continue  # dinero dentro de caja es lo esperado
            self._caja(frame, x1, y1, x2, y2, (0, 0, 255), grosor=1)
            self._etiqueta(frame, "BILLETE FUERA DE CAJA", x1, y1, (0, 0, 255))
            self._emitir_evento(
                "billete_fuera_de_caja",
                0,
                {"alerta": "revisar_manualmente", "confianza": round(float(box.conf[0]), 2)},
                clave_evento=f"billete_fuera_de_caja:{cx // 50}_{cy // 50}",
            )

    # ── Helpers ──────────────────────────────────────────────────────────

    def _identificar_zonas(self):
        """Zona caja: por nombre, o la 1ra zona dibujada si ninguna coincide.
        Zona restringida: SOLO por nombre — sin respaldo posicional, para no
        activar una regla sensible por una zona configurada para otra cosa."""
        nombres_zonas = list(self.zonas.keys())

        nombre_restringida = next(
            (n for n in nombres_zonas if any(p in n.lower() for p in PALABRAS_ZONA_RESTRINGIDA)),
            None,
        )
        nombre_caja = next(
            (n for n in nombres_zonas if any(p in n.lower() for p in PALABRAS_ZONA_CAJA)),
            None,
        )
        if not nombre_caja:
            candidatas = [n for n in nombres_zonas if n != nombre_restringida]
            nombre_caja = candidatas[0] if candidatas else None

        return nombre_caja, nombre_restringida

    @staticmethod
    def _pct_a_px(zona_pct, ancho, alto):
        return {
            "x1": int(zona_pct["x1"] / 100.0 * ancho),
            "y1": int(zona_pct["y1"] / 100.0 * alto),
            "x2": int(zona_pct["x2"] / 100.0 * ancho),
            "y2": int(zona_pct["y2"] / 100.0 * alto),
        }

    @staticmethod
    def _dentro(cx, cy, zona):
        return zona["x1"] < cx < zona["x2"] and zona["y1"] < cy < zona["y2"]

    def _obtener_o_crear(self, almacen, atributo_next_id, cx, cy, x1, y1, x2, y2, ahora):
        """Logica de proximidad (<50px) para mantener el ID de una persona/objeto entre frames."""
        for e_id, data in almacen.items():
            if math.hypot(cx - data["cx"], cy - data["cy"]) < 50:
                data.update(cx=cx, cy=cy, x1=x1, y1=y1, x2=x2, y2=y2, visto=ahora)
                return e_id

        e_id = getattr(self, atributo_next_id)
        setattr(self, atributo_next_id, e_id + 1)
        almacen[e_id] = {
            "cx": cx, "cy": cy, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "inicio": ahora, "visto": ahora, "visto_anterior": ahora,
        }
        return e_id

    @staticmethod
    def _etiqueta(frame, texto, x, y, color_fondo, color_texto=(255, 255, 255)):
        (w_txt, h_txt), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x, y - h_txt - 8), (x + w_txt + 10, y), color_fondo, -1)
        cv2.putText(frame, texto, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_texto, 1, cv2.LINE_AA)
