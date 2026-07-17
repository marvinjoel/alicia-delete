import time
import math
import cv2
from processors.base_processor import BaseProcessor

# Objetos de consumo que YOLO/COCO ya reconoce y que indican que el cliente
# recibio algo (senal de compra). "dinero"/"tarjeta"/"recibo" NO son clases
# del modelo base — detectarlos requeriria un modelo entrenado a medida u OCR
# (ver processors/format_change_tracker.py para un ejemplo de OCR con Tesseract).
CLASES_COMPRA = {"cup", "bottle", "wine glass", "bowl"}

CONFIANZA_MINIMA = 0.5

# Si el nombre de una zona (definido en el panel) contiene alguna de estas
# palabras, se usa como zona de personal sin importar el orden en que se dibujo.
PALABRAS_ZONA_PERSONAL = ("personal", "staff", "empleado", "mesero", "cajero")


class AbandonedSaleTrackerProcessor(BaseProcessor):
    """
    Rastrea clientes potenciales en la zona de atencion (barra/caja) y detecta
    si se retiran sin haber comprado: sin haber interactuado con el personal
    ni mostrado intencion de compra (CLASES_COMPRA).

    Zonas configuradas desde el panel (config_json.zonas, % del frame, mismo
    formato que efficiency_tracker/people_counter). La zona de personal se
    identifica por nombre (ver PALABRAS_ZONA_PERSONAL, ej. "Zona Personal",
    "Cajero"); si ninguna zona coincide por nombre, se usa la 2da zona dibujada
    como respaldo. La zona restante es siempre la zona de atencion (obligatoria).
    Sin zonas configuradas, el modulo no evalua nada (no usa valores de pixel
    "por defecto" que no corresponderian al encuadre real de cada camara).
    """

    def __init__(self, camara_id, config):
        super().__init__(camara_id, config)
        self.zonas = self.config_extra.get("zonas", {})
        self.tiempo_interes_seg    = float(self.config_extra.get("tiempo_interes", 5.0))
        self.distancia_objeto_px   = float(self.config_extra.get("distancia_objeto_px", 100))
        self.distancia_personal_px = float(self.config_extra.get("distancia_personal_px", 150))
        # Segundos que se retiene la memoria de un cliente tras perderlo de vista
        # en un frame (oclusion, parpadeo de deteccion) antes de darlo por retirado.
        # Sin esto, un solo frame sin deteccion crea un ID nuevo en vez de seguir
        # con el mismo cliente.
        self.gracia_seg = float(self.config_extra.get("gracia_seg", 3.0))

        # cliente_id -> {"cx","cy","x1","y1","x2","y2","inicio","visto","interactuo","motivo"}
        self.clientes_en_zona = {}
        self.next_id = 1

    def procesar(self, frame, resultados):
        if not self.zonas:
            if self._puede_loguear():
                print(f"[{self.modulo}] Cam={self.camara_id} | sin zona de atencion configurada, omitiendo")
            return frame

        alto, ancho = frame.shape[:2]
        nombre_atencion, nombre_personal = self._identificar_zonas()
        zona_atencion = self._pct_a_px(self.zonas[nombre_atencion], ancho, alto)
        zona_personal = self._pct_a_px(self.zonas[nombre_personal], ancho, alto) if nombre_personal else None

        nombres = resultados[0].names
        cajas   = resultados[0].boxes
        ahora   = time.time()

        self._caja(frame, zona_atencion["x1"], zona_atencion["y1"], zona_atencion["x2"], zona_atencion["y2"], color=(255, 0, 255), grosor=1)
        self._etiqueta(frame, "ATENCION", zona_atencion["x1"], zona_atencion["y1"], (255, 0, 255))
        if zona_personal:
            self._caja(frame, zona_personal["x1"], zona_personal["y1"], zona_personal["x2"], zona_personal["y2"], color=(255, 144, 30), grosor=1)
            self._etiqueta(frame, nombre_personal.upper(), zona_personal["x1"], zona_personal["y1"], (255, 144, 30))

        personas       = []
        objetos_compra = []
        personal_pos   = []

        for box in cajas:
            if float(box.conf[0]) < CONFIANZA_MINIMA:
                continue
            clase_nom = nombres[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if clase_nom == "person":
                if zona_personal and self._dentro(cx, cy, zona_personal):
                    personal_pos.append((cx, cy))
                else:
                    personas.append((cx, cy, x1, y1, x2, y2))
            elif clase_nom in CLASES_COMPRA:
                objetos_compra.append((cx, cy))

        # 1. Actualizar/crear clientes dentro de la zona de atencion
        clientes_actuales = set()
        for cx, cy, x1, y1, x2, y2 in personas:
            if not self._dentro(cx, cy, zona_atencion):
                continue

            cliente_id = self._obtener_o_crear_id(cx, cy, x1, y1, x2, y2, ahora)
            clientes_actuales.add(cliente_id)
            data = self.clientes_en_zona[cliente_id]

            if not data["interactuo"]:
                cerca_objeto = any(
                    math.hypot(cx - ox, cy - oy) < self.distancia_objeto_px
                    for ox, oy in objetos_compra
                )
                cerca_personal = any(
                    math.hypot(cx - px_, cy - py_) < self.distancia_personal_px
                    for px_, py_ in personal_pos
                )
                if cerca_objeto or cerca_personal:
                    data["interactuo"] = True
                    data["motivo"] = "objeto_consumo" if cerca_objeto else "personal"

            self._dibujar_cliente(frame, cliente_id, data, ahora)

        # 2. Detectar abandonos reales: clientes sin deteccion desde hace mas de
        # gracia_seg (un solo frame perdido no cuenta como abandono real)
        for c_id in list(self.clientes_en_zona.keys()):
            if c_id in clientes_actuales:
                continue
            data = self.clientes_en_zona[c_id]
            if ahora - data["visto"] < self.gracia_seg:
                continue  # posible parpadeo de deteccion, esperar

            del self.clientes_en_zona[c_id]
            duracion = data["visto"] - data["inicio"]

            if duracion >= self.tiempo_interes_seg and not data["interactuo"]:
                self._emitir_evento(
                    "abandono_sin_compra",
                    int(duracion),
                    {
                        "cliente_id": c_id,
                        "tiempo_permanencia": round(duracion, 1),
                    },
                    # Cooldown por cliente (no por camara): sin esto, dos clientes
                    # distintos que abandonan dentro del mismo ALERTA_COOLDOWN_MIN
                    # pisarian el evento del segundo.
                    clave_evento=f"abandono_sin_compra:{c_id}",
                )

        if self._puede_loguear():
            print(f"[{self.modulo}] Cam={self.camara_id} | clientes en zona={len(clientes_actuales)}")

        return frame

    # ── Helpers ──────────────────────────────────────────────────────────

    def _identificar_zonas(self):
        """Devuelve (nombre_zona_atencion, nombre_zona_personal | None).
        La zona de personal se busca primero por nombre (PALABRAS_ZONA_PERSONAL);
        si ninguna coincide, se usa la 2da zona dibujada como respaldo posicional."""
        nombres_zonas = list(self.zonas.keys())

        nombre_personal = next(
            (n for n in nombres_zonas if any(p in n.lower() for p in PALABRAS_ZONA_PERSONAL)),
            None,
        )
        candidatas_atencion = [n for n in nombres_zonas if n != nombre_personal]
        nombre_atencion = candidatas_atencion[0] if candidatas_atencion else nombres_zonas[0]

        if not nombre_personal:
            candidatas_personal = [n for n in nombres_zonas if n != nombre_atencion]
            nombre_personal = candidatas_personal[0] if candidatas_personal else None

        return nombre_atencion, nombre_personal

    @staticmethod
    def _pct_a_px(zona_pct, ancho, alto):
        """Convierte una zona en % (formato del panel) a coordenadas de pixel del frame actual."""
        return {
            "x1": int(zona_pct["x1"] / 100.0 * ancho),
            "y1": int(zona_pct["y1"] / 100.0 * alto),
            "x2": int(zona_pct["x2"] / 100.0 * ancho),
            "y2": int(zona_pct["y2"] / 100.0 * alto),
        }

    @staticmethod
    def _dentro(cx, cy, zona):
        return zona["x1"] < cx < zona["x2"] and zona["y1"] < cy < zona["y2"]

    def _obtener_o_crear_id(self, cx, cy, x1, y1, x2, y2, ahora):
        """Logica simple de proximidad (<50px entre frames) para mantener el ID."""
        for c_id, data in self.clientes_en_zona.items():
            if math.hypot(cx - data["cx"], cy - data["cy"]) < 50:
                data.update(cx=cx, cy=cy, x1=x1, y1=y1, x2=x2, y2=y2, visto=ahora)
                return c_id

        c_id = self.next_id
        self.next_id += 1
        self.clientes_en_zona[c_id] = {
            "cx": cx, "cy": cy, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "inicio": ahora, "visto": ahora, "interactuo": False, "motivo": None,
        }
        return c_id

    @staticmethod
    def _etiqueta(frame, texto, x, y, color_fondo, color_texto=(0, 0, 0)):
        """Etiqueta compacta con fondo solido (sin el contorno grueso de _texto)."""
        (w_txt, h_txt), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x, y - h_txt - 8), (x + w_txt + 10, y), color_fondo, -1)
        cv2.putText(frame, texto, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_texto, 1, cv2.LINE_AA)

    def _dibujar_cliente(self, frame, cliente_id, data, ahora):
        duracion = ahora - data["inicio"]
        color = (0, 200, 0) if data["interactuo"] else (0, 0, 220)
        self._caja(frame, data["x1"], data["y1"], data["x2"], data["y2"], color, grosor=1)
        etiqueta = f"Cliente {cliente_id} | {self._fmt_tiempo(duracion)}"
        if data["interactuo"]:
            etiqueta += f" | {data['motivo']}"
        self._etiqueta(frame, etiqueta, data["x1"], data["y1"], color, (255, 255, 255))
