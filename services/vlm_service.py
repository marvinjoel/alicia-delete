"""
Servicio de Vision IA — FastVLM (Apple)
Analiza frames del video y genera descripciones en español.
Corre en un thread separado por camara, sin bloquear YOLO.
"""
import os, time, threading, asyncio
import cv2
from PIL import Image

# ── Modelos FastVLM disponibles ────────────────────────────────────────────
MODELOS = {
    "FastVLM-0.5B": "apple/FastVLM-0.5B",   # Muy rapido  (~1 GB RAM)
    "FastVLM-1.5B": "apple/FastVLM-1.5B",   # Balanceado  (~3 GB RAM)
    "FastVLM-7B":   "apple/FastVLM-7B",      # Mas potente (~14 GB RAM)
}

# Prompt accion-focused: pregunta "que esta pasando" (mejor para vigilancia que "describe").
PROMPT_ESCENA = "What is happening in this image? Answer in 1 short sentence. Only observable facts."

IMAGE_TOKEN_INDEX = -200  # Token especial de FastVLM para <image>


class VLMService:
    """Singleton — carga un modelo FastVLM y analiza frames periodicamente."""
    _inst = None

    def __new__(cls):
        if not cls._inst:
            cls._inst = super().__new__(cls)
            cls._inst._ok = False
        return cls._inst

    def __init__(self):
        if self._ok:
            return
        self._ok = True
        self.modelo = None          # modelo cargado (AutoModelForCausalLM)
        self.tokenizer = None       # tokenizer del modelo
        self.img_processor = None   # procesador de imagen (del vision tower)
        self.nombre = None          # nombre legible ("FastVLM-0.5B")
        self.device = None          # "cuda" o "cpu"
        self.cargando = False
        self._lock = threading.Lock()
        self._cams = {}             # cam_id -> {"activo": bool, "intervalo": int}
        self._crear_tabla()

    # ── Crear tabla BD si no existe ────────────────────────────────────────
    def _crear_tabla(self):
        try:
            from services.database import get_connection
            conn = get_connection()
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS vision_ia_log (
                        rowid        INT AUTO_INCREMENT PRIMARY KEY,
                        fk_camara    INT NOT NULL,
                        modelo       VARCHAR(50) NOT NULL,
                        descripcion  TEXT NOT NULL,
                        creado_fecha DATETIME DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_cam_fecha (fk_camara, creado_fecha)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci
                """)
            conn.close()
        except Exception as e:
            print(f"[VLM] Aviso tabla: {e}")

    # ── Carga de modelo (en background) ────────────────────────────────────
    def cargar(self, nombre="FastVLM-0.5B", cam_id=None, intervalo=10):
        """Inicia carga del modelo en un thread aparte."""
        if self.cargando:
            return
        self.cargando = True
        threading.Thread(
            target=self._cargar_bg, args=(nombre, cam_id, intervalo), daemon=True
        ).start()

    def _cargar_bg(self, nombre, cam_id, intervalo):
        """Descarga y carga el modelo FastVLM. Tarda la primera vez."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            model_id = MODELOS.get(nombre, MODELOS["FastVLM-0.5B"])
            # IA_DEVICE en config.env: "auto" | "cuda" | "cpu"
            dev_cfg = os.getenv("IA_DEVICE", "auto").lower()
            if dev_cfg == "cpu":
                self.device = "cpu"
            elif dev_cfg == "cuda":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:  # auto
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if self.device == "cuda" else torch.float32

            print(f"[VLM] Cargando {nombre} ({model_id}) en {self.device}...")

            # FastVLM no tiene preprocessor_config.json —
            # el image_processor se obtiene del vision tower del modelo
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True
            )
            self.modelo = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=dtype, trust_remote_code=True
            ).to(self.device).eval()

            # Obtener el procesador de imagen desde el vision tower
            self.img_processor = self.modelo.get_vision_tower().image_processor

            self.nombre = nombre
            print(f"[VLM] {nombre} cargado OK en {self.device}")

            # Notificar via WebSocket
            self._broadcast_estado(cam_id, "listo", nombre)

            # Auto-iniciar camara si se pidio
            if cam_id:
                self.iniciar_camara(cam_id, intervalo)

        except Exception as e:
            print(f"[VLM] Error cargando: {e}")
            self.modelo = None
            self.tokenizer = None
            self.img_processor = None
            self._broadcast_estado(cam_id, "error", str(e))
        finally:
            self.cargando = False

    # ── Inferencia ─────────────────────────────────────────────────────────
    def describir(self, imagen_pil: Image.Image):
        """Genera descripcion periodica de la escena (prompt fijo)."""
        return self.responder(imagen_pil, PROMPT_ESCENA)

    def responder(self, imagen_pil: Image.Image, pregunta: str, max_tokens: int = None):
        # Si no se pasa max_tokens, usar el de config.env
        if max_tokens is None:
            max_tokens = int(os.getenv("VLM_MAX_TOKENS", "220"))
        """Genera respuesta es español a una pregunta sobre una imagen. Thread-safe.
        Usado para analisis automatico y para chat en tiempo real."""
        if not self.modelo or not self.tokenizer or not self.img_processor:
            return None
        import torch
        with self._lock:
            try:
                # Redimensionar imagen grande (mejor inferencia y mas rapido)
                max_lado = 640
                if max(imagen_pil.size) > max_lado:
                    imagen_pil.thumbnail((max_lado, max_lado))

                # Construir prompt con <image> usando chat template
                msgs = [{"role": "user", "content": f"<image>\n{pregunta}"}]
                rendered = self.tokenizer.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=False
                )
                print(f"[VLM DEBUG] prompt rendered: {repr(rendered[:200])}")

                # Separar texto antes y despues de <image>
                pre, post = rendered.split("<image>", 1)
                pre_ids  = self.tokenizer(pre,  return_tensors="pt", add_special_tokens=False).input_ids
                post_ids = self.tokenizer(post, return_tensors="pt", add_special_tokens=False).input_ids

                # Insertar token de imagen (-200) entre pre y post
                img_tok   = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
                input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(self.device)
                attn_mask = torch.ones_like(input_ids, device=self.device)

                # Procesar imagen via el vision tower del modelo
                px = self.img_processor(
                    images=imagen_pil, return_tensors="pt"
                )["pixel_values"].to(self.device, dtype=self.modelo.dtype)

                print(f"[VLM DEBUG] input_ids.shape={input_ids.shape}, px.shape={px.shape}, max_tokens={max_tokens}")

                # Generar respuesta (repetition_penalty evita loops de "I'm sorry...")
                with torch.no_grad():
                    ids = self.modelo.generate(
                        inputs=input_ids,
                        attention_mask=attn_mask,
                        images=px,
                        max_new_tokens=max_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                        min_new_tokens=10,
                        repetition_penalty=1.3,  # penaliza repetir palabras/frases
                        no_repeat_ngram_size=4,  # no repite secuencias de 4+ palabras
                    )

                # Decodificar solo tokens nuevos
                print(f"[VLM DEBUG] output ids.shape={ids.shape}, nuevos={ids.shape[1] - input_ids.shape[1]}")
                nuevos = ids[0][input_ids.shape[-1]:]

                # DEBUG: decodificar SIN saltar tokens especiales
                texto_raw = self.tokenizer.decode(nuevos, skip_special_tokens=False).strip()
                print(f"[VLM RAW_FULL] {texto_raw[:300]}")

                texto = self.tokenizer.decode(nuevos, skip_special_tokens=True).strip()
                print(f"[VLM RAW] {texto[:200]}")

                # ── Limpieza profesional en pasos ──────────────────────────
                texto = self._limpiar_respuesta(texto)

                # ── Traducir al español si la bandera esta activa ──────────
                if texto and os.getenv("VLM_TRADUCIR", "false").lower() == "true":
                    t_trad = self._traducir_espanol(texto)
                    if t_trad:
                        texto = t_trad

                return texto

            except Exception as e:
                print(f"[VLM] Error inferencia: {e}")
                return None

    def obtener_frame_pil(self, cam_id: int):
        """Obtiene el frame actual de la camara como PIL Image."""
        from workers.camera_worker import workers_activos
        worker = workers_activos.get(cam_id)
        if not worker or worker.frame_raw is None:
            return None
        return Image.fromarray(cv2.cvtColor(worker.frame_raw, cv2.COLOR_BGR2RGB))

    # ── Traducir al español via OpenAI (rapido: ~0.3s) ─────────────────────
    @staticmethod
    def _traducir_espanol(texto_en: str) -> str:
        """Traduce de ingles a español via gpt-4o-mini. Si falla, devuelve vacio."""
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            return ""
        try:
            import httpx
            modelo = os.getenv("OPENAI_MODELO", "gpt-4o-mini")
            r = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": modelo,
                    "messages": [
                        {"role": "system", "content": "Traduce al español. Devuelve SOLO la traduccion, sin comentarios ni prefijos."},
                        {"role": "user", "content": texto_en},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 200,
                },
                timeout=8.0,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[VLM] Error traduccion: {e}")
            return ""

    # ── Limpieza de respuesta del modelo ───────────────────────────────────
    @staticmethod
    def _limpiar_respuesta(texto: str) -> str:
        """Extrae la respuesta util, descartando prefijos y meta-texto."""
        import re
        if not texto:
            return ""

        # 1) Quitar prefijos tipo "Answer:" o "**Answer:**" al inicio
        texto = re.sub(
            r"^\s*\**\s*(Answer|Respuesta|Description|Descripcion|Descripción)\s*:?\s*\**\s*",
            "", texto, flags=re.IGNORECASE,
        ).strip()

        # 2) Cortar al primer marcador de meta-texto
        MARCADORES = [
            "I'm sorry", "I cannot", "Sorry,", "I am sorry", "Lo siento", "No puedo",
            "**Question:**", "Question:", "Pregunta:", "**Pregunta:**",
            "This task", "This description", "This image aims", "The question seems",
            "I hope this", "Please let me know", "If you have", "Feel free",
            "<end of", "<start of", "No ethical", "This does not",
            "Note:", "Observacion:",
        ]
        for marca in MARCADORES:
            idx = texto.lower().find(marca.lower())
            if idx >= 0:
                texto = texto[:idx].strip()

        # 3) Cortar en la primera oracion completa (min 20 chars).
        # El punto debe ir precedido por LETRA (no digito) para no cortar en "1." "2." etc.
        match = re.search(r"(?<=[a-zA-Z])[.!?](?:\s|$)", texto[20:])
        if match:
            texto = texto[:20 + match.start() + 1].strip()

        # 4) Si no termina en puntuacion, cortar hasta la ultima completa
        if texto and texto[-1] not in ".!?":
            ultimo = max(texto.rfind(". "), texto.rfind("! "), texto.rfind("? "))
            if ultimo > 20:
                texto = texto[:ultimo + 1].strip()

        # 5) Capitalizar primera letra
        if texto and texto[0].islower():
            texto = texto[0].upper() + texto[1:]

        # 6) Solo descartar si esta MUY corto (probable basura)
        if len(texto) < 5:
            return ""

        return texto

    # ── Control por camara ─────────────────────────────────────────────────
    def iniciar_camara(self, cam_id: int, intervalo: int = 10):
        """Inicia analisis periodico para una camara."""
        if cam_id in self._cams and self._cams[cam_id]["activo"]:
            return
        self._cams[cam_id] = {"activo": True, "intervalo": intervalo}
        threading.Thread(
            target=self._loop, args=(cam_id,), daemon=True, name=f"vlm-{cam_id}"
        ).start()
        self._broadcast_estado(cam_id, "activo", self.nombre or "")

    def detener_camara(self, cam_id: int):
        """Detiene el analisis para una camara."""
        if cam_id in self._cams:
            self._cams[cam_id]["activo"] = False
            self._broadcast_estado(cam_id, "detenido")

    def _loop(self, cam_id: int):
        """Loop de analisis: toma frame -> describe -> guarda -> broadcast."""
        from workers.camera_worker import workers_activos
        from services.database import insertar_vision_log

        print(f"[VLM] Analisis activo cam {cam_id}")
        contador = 0

        while self._cams.get(cam_id, {}).get("activo"):
            intervalo = self._cams[cam_id]["intervalo"]
            worker = workers_activos.get(cam_id)

            if worker and worker.frame_raw is not None:
                try:
                    contador += 1
                    t0 = time.time()
                    print(f"[VLM] cam {cam_id} #{contador} - analizando frame ({worker.frame_raw.shape[1]}x{worker.frame_raw.shape[0]})...")

                    # BGR -> RGB -> PIL
                    img = Image.fromarray(
                        cv2.cvtColor(worker.frame_raw, cv2.COLOR_BGR2RGB)
                    )
                    desc = self.describir(img)
                    elapsed = time.time() - t0

                    if desc:
                        # Preview: primeras 80 chars
                        preview = desc[:80] + ("..." if len(desc) > 80 else "")
                        print(f"[VLM] cam {cam_id} #{contador} OK ({elapsed:.1f}s): {preview}")
                        # 1) Enviar por WebSocket PRIMERO (para que el front lo vea ya)
                        self._broadcast_vision(cam_id, desc)
                        # 2) Guardar en BD despues (no bloquea al front)
                        insertar_vision_log(cam_id, self.nombre or "FastVLM", desc)
                    else:
                        print(f"[VLM] cam {cam_id} #{contador} vacio ({elapsed:.1f}s)")
                except Exception as e:
                    print(f"[VLM] Error cam {cam_id}: {e}")
            else:
                print(f"[VLM] cam {cam_id} sin frame disponible (esperando...)")

            time.sleep(intervalo)

        print(f"[VLM] Analisis detenido cam {cam_id}")

    # ── Helpers WebSocket ──────────────────────────────────────────────────
    def _broadcast_vision(self, cam_id, descripcion):
        """Envia descripcion por WebSocket."""
        self._ws_send(cam_id, {
            "tipo": "vision_ia",
            "camara_id": cam_id,
            "modelo": self.nombre,
            "descripcion": descripcion,
            "timestamp": time.strftime("%H:%M:%S"),
        })

    def _broadcast_estado(self, cam_id, estado, info=""):
        """Envia cambio de estado VLM por WebSocket."""
        if cam_id:
            self._ws_send(cam_id, {
                "tipo": "vlm_estado",
                "estado": estado,
                "modelo": self.nombre or info,
                "device": self.device,
                "info": info,
            })

    def _ws_send(self, cam_id, data):
        """Envia data por WebSocket desde un thread."""
        try:
            from api.routes_ws import ws_manager, _main_loop
            if _main_loop:
                asyncio.run_coroutine_threadsafe(
                    ws_manager.broadcast(cam_id, data), _main_loop
                )
        except Exception:
            pass


# Instancia global
vlm = VLMService()
