import cv2
import os
import time
import threading

from ultralytics import YOLO
from services.database import get_camara, get_analiticas_camara
# from services.database import insertar_snapshot  # snapshot desactivado — usando VLM en navegador

from processors.dirty_tables       import DirtyTableProcessor
from processors.table_occupancy    import TableOccupancyProcessor
from processors.customer_wait_time import CustomerWaitTimeProcessor
from processors.staff_tracker      import StaffTrackerProcessor
from processors.people_counter     import PeopleCounterProcessor
from processors.queue_detector     import QueueDetectorProcessor
from processors.lego_tracker       import LegoTrackerProcessor

PROCESADORES_MAP = {
    "dirty_table_detector": DirtyTableProcessor,
    "table_occupancy":      TableOccupancyProcessor,
    "customer_wait_time":   CustomerWaitTimeProcessor,
    "staff_tracker":        StaffTrackerProcessor,
    "people_counter":       PeopleCounterProcessor,
    "queue_detector":       QueueDetectorProcessor,
    "lego_tracker":         LegoTrackerProcessor,
}

# Registro global: camara_id -> CameraWorker
workers_activos: dict[int, "CameraWorker"] = {}


class CameraWorker:
    """
    Worker por camara. Corre en un thread daemon.
    Lee frames con OpenCV, los procesa con YOLO,
    ejecuta los procesadores activos y guarda el
    ultimo frame anotado en self.frame_actual (bytes JPEG).
    """

    def __init__(self, camara_id: int):
        self.camara_id    = camara_id
        self.corriendo    = False
        self.frame_actual = None   # bytes JPEG del ultimo frame
        self.frame_raw    = None   # Frame numpy BGR sin anotar, para Vision IA
        self._thread      = None

        # ── Cargar camara desde BD ─────────────────────────────────────────
        camara = get_camara(camara_id)
        if not camara:
            raise ValueError(f"Camara {camara_id} no encontrada en la base de datos")

        self.url_stream = camara["url_stream"]
        if not self.url_stream:
            raise ValueError(f"Camara {camara_id} no tiene URL de stream configurada")

        # ── Cargar analiticas activas ──────────────────────────────────────
        analiticas = get_analiticas_camara(camara_id)
        if not analiticas:
            raise ValueError(f"Camara {camara_id} no tiene modulos de analitica activos")

        # ── Cargar modelo YOLO ─────────────────────────────────────────────
        # MODELO_YOLO_DEFAULT en config.env sobreescribe el valor de la BD.
        # Util para usar un modelo ligero en local y uno preciso en produccion.
        modelo_yolo = (
            os.getenv("MODELO_YOLO_DEFAULT")
            or analiticas[0].get("modelo_yolo", "yolo11m.pt")
        )
        print(f"[Worker {camara_id}] Cargando modelo: {modelo_yolo}")
        self.modelo = YOLO(f"models/{modelo_yolo}")

        # Mover YOLO al dispositivo configurado (IA_DEVICE en config.env)
        try:
            import torch
            dev_cfg = os.getenv("IA_DEVICE", "auto").lower()
            if dev_cfg == "cpu":
                yolo_device = "cpu"
            else:  # auto o cuda
                yolo_device = "cuda" if torch.cuda.is_available() else "cpu"
            self.modelo.to(yolo_device)
            print(f"[Worker {camara_id}] YOLO en {yolo_device}")
        except Exception as e:
            print(f"[Worker {camara_id}] Aviso device YOLO: {e}")

        # ── Instanciar procesadores ────────────────────────────────────────
        # Lista de procesadores a desactivar desde config.env
        desactivados = set(
            m.strip() for m in os.getenv("PROCESADORES_DESACTIVAR", "").split(",") if m.strip()
        )

        self.procesadores = []
        for an in analiticas:
            modulo = an["modulo"]
            if modulo in desactivados:
                print(f"[Worker {camara_id}] Modulo DESACTIVADO (config.env): {modulo}")
                continue
            clase = PROCESADORES_MAP.get(modulo)
            if clase:
                self.procesadores.append(clase(camara_id, an))
                print(f"[Worker {camara_id}] Modulo activo: {modulo}")
            else:
                print(f"[Worker {camara_id}] Modulo desconocido ignorado: {modulo}")

    # ── Control ────────────────────────────────────────────────────────────

    def iniciar(self):
        self.corriendo = True
        self._thread   = threading.Thread(target=self._loop, daemon=True, name=f"cam-{self.camara_id}")
        self._thread.start()
        print(f"[Worker {self.camara_id}] Thread iniciado")

    def detener(self):
        self.corriendo    = False
        self.frame_actual = None
        self.frame_raw    = None
        print(f"[Worker {self.camara_id}] Deteniendo...")

    # ── Loop principal ─────────────────────────────────────────────────────

    def _loop(self):
        from services.redis_client import set_camara_activa, set_camara_inactiva
        set_camara_activa(self.camara_id)

        # Si la URL es un numero (ej: "0"), convertir a int para webcam/USB
        fuente = int(self.url_stream) if self.url_stream.strip().isdigit() else self.url_stream

        # ── Modo webcam del navegador ──────────────────────────────────────
        # Los frames llegan via POST /webcam/frame — este worker solo queda
        # registrado como "activo" pero no abre ninguna camara.
        if fuente == 'browser':
            print(f"[Worker {self.camara_id}] Modo webcam navegador — frames via /webcam/frame")
            while self.corriendo:
                time.sleep(1)
            set_camara_inactiva(self.camara_id)
            return

        # ── Resolver YouTube / youtu.be → URL directa CDN ─────────────────
        if isinstance(fuente, str) and ('youtube.com' in fuente or 'youtu.be' in fuente):
            print(f"[Worker {self.camara_id}] Resolviendo URL de YouTube...")
            try:
                import yt_dlp
                ydl_opts = {
                    'format': 'best[protocol^=m3u8]/best[height<=720]/best',
                    'quiet': True,
                    'no_warnings': True,
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info   = ydl.extract_info(fuente, download=False)
                    fuente = info['url']
                print(f"[Worker {self.camara_id}] CDN URL obtenida OK")
            except Exception as e:
                raise ValueError(f"No se pudo resolver YouTube: {e}")

        # Procesar 1 de cada N frames con YOLO (el resto solo se lee para vaciar el buffer)
        PROCESAR_CADA_N = int(os.getenv("PROCESAR_CADA_N", "3"))
        # Ancho maximo antes de pasar a YOLO (0 = sin resize)
        YOLO_MAX_WIDTH   = int(os.getenv("YOLO_MAX_WIDTH", "640"))

        def _abrir_cap(f):
            if isinstance(f, str) and f.startswith("http"):
                # MJPEG sobre HTTP requiere backend FFMPEG en Windows
                c = cv2.VideoCapture(f, cv2.CAP_FFMPEG)
            else:
                c = cv2.VideoCapture(f)
            # Buffer minimo: evita acumular frames viejos en streams lentos
            c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return c

        print(f"[Worker {self.camara_id}] Conectando a: {fuente}")
        cap = _abrir_cap(fuente)

        intentos_reconexion = 0
        contador_frames     = 0

        # ── Acumuladores para snapshot (desactivado — usando VLM en navegador) ──
        # SNAPSHOT_INTERVALO  = 60
        # ultimo_snapshot     = time.time()
        # acum_personas       = []
        # acum_objetos        = {}
        # acum_mesas_sucias   = []
        # _CLASES_SUCIEDAD = {"cup","bowl","bottle","fork","knife","spoon","wine glass","plate"}

        while self.corriendo:
            ok, frame = cap.read()

            if not ok:
                intentos_reconexion += 1
                print(f"[Worker {self.camara_id}] Sin frame ({intentos_reconexion}). Reconectando...")
                time.sleep(2)
                cap.release()
                cap = _abrir_cap(fuente)
                continue

            intentos_reconexion  = 0
            contador_frames     += 1

            # Saltar frames: leer todos para vaciar buffer, pero YOLO solo en 1 de N
            if contador_frames % PROCESAR_CADA_N != 0:
                continue

            try:
                # Guardar frame limpio para Vision IA (VLM)
                self.frame_raw = frame

                # ── Resize opcional antes de YOLO (reduce tiempo de inferencia) ──
                frame_yolo = frame
                if YOLO_MAX_WIDTH > 0 and frame.shape[1] > YOLO_MAX_WIDTH:
                    escala     = YOLO_MAX_WIDTH / frame.shape[1]
                    nuevo_alto = int(frame.shape[0] * escala)
                    frame_yolo = cv2.resize(frame, (YOLO_MAX_WIDTH, nuevo_alto))

                # ── YOLO: detectar objetos en el frame ─────────────────────
                resultados = self.modelo(frame_yolo, verbose=False)

                # Dibujar cajas YOLO en el frame
                frame_out = resultados[0].plot()  # ← paso con deteccion de otros obj
                #frame_out = frame                  # ← pasar frame limpio

                # ── [snapshot desactivado — VLM analiza frames directamente en browser] ──

                # ── Procesadores: logica de negocio + overlays personalizados
                for proc in self.procesadores:
                    frame_out = proc.procesar(frame_out, resultados)

                # ── Convertir a JPEG y guardar en memoria ──────────────────
                _, jpeg = cv2.imencode(
                    ".jpg", frame_out,
                    [cv2.IMWRITE_JPEG_QUALITY, 72]
                )
                self.frame_actual = jpeg.tobytes()

            except Exception as e:
                print(f"[Worker {self.camara_id}] Error procesando frame: {e}")
                continue

        cap.release()
        set_camara_inactiva(self.camara_id)
        print(f"[Worker {self.camara_id}] Detenido correctamente.")
