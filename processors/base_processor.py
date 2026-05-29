import os
import cv2
import time
from abc import ABC, abstractmethod


class BaseProcessor(ABC):
    """
    Clase base para todos los modulos de analitica.
    Cada procesador recibe un frame y los resultados de YOLO,
    dibuja overlays y emite eventos cuando corresponde.
    """

    def __init__(self, camara_id: int, config: dict):
        self.camara_id      = camara_id
        self.modulo         = config.get("modulo", "")
        self.umbral_seg     = float(config.get("umbral_minutos", 2)) * 60
        self.config_extra   = config.get("config_json") or {}
        # Cooldown global: evita flood de BD (min 30s entre eventos del mismo tipo)
        self._ultimo_evento   = {}  # tipo_evento -> timestamp ultima insercion
        # Cooldown leido de config.env: ALERTA_COOLDOWN_MIN (minutos) → segundos
        self._evento_cooldown = float(os.getenv("ALERTA_COOLDOWN_MIN", "1")) * 60
        # Throttle de logs: maximo 1 log cada 3 segundos por procesador
        self._ultimo_log      = 0.0
        print(f"[{self.modulo}] Cam={camara_id} | umbral={self.umbral_seg}s | config={self.config_extra}")

    @abstractmethod
    def procesar(self, frame, resultados):
        """
        Analiza el frame y los resultados YOLO.
        Dibuja overlays sobre el frame.
        Retorna el frame modificado.
        """
        pass

    # ── Helpers de dibujo ──────────────────────────────────────────────────

    def _texto(self, frame, texto, pos, color=(0, 0, 255), escala=0.65):
        cv2.putText(frame, texto, pos, cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), 3)
        cv2.putText(frame, texto, pos, cv2.FONT_HERSHEY_SIMPLEX, escala, color, 2)

    def _caja(self, frame, x1, y1, x2, y2, color=(0, 0, 255), grosor=2):
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, grosor)

    def _panel(self, frame, texto, x, y, color_fondo=(0, 0, 180)):
        """Panel semitransparente con texto en una esquina."""
        overlay = frame.copy()
        w = len(texto) * 11 + 20
        cv2.rectangle(overlay, (x, y), (x + w, y + 34), color_fondo, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, texto, (x + 8, y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    def _puede_loguear(self, intervalo: float = 3.0) -> bool:
        """Retorna True si han pasado `intervalo` segundos desde el ultimo log.
        Actualiza el timestamp internamente. Usar en if: if self._puede_loguear(): ..."""
        ahora = time.time()
        if ahora - self._ultimo_log >= intervalo:
            self._ultimo_log = ahora
            return True
        return False

    def _fmt_tiempo(self, segundos: float) -> str:
        m = int(segundos // 60)
        s = int(segundos % 60)
        return f"{m}:{s:02d}"

    # ── Emision de eventos ─────────────────────────────────────────────────

    def _emitir_evento(self, tipo_evento: str, duracion_seg: int, datos: dict):
        """
        Guarda el evento en BD con cooldown por tipo_evento (30s por defecto).
        Siempre hace broadcast por WebSocket al dashboard.
        """
        import time
        from services.database import insertar_evento

        ahora = time.time()
        if ahora - self._ultimo_evento.get(tipo_evento, 0) < self._evento_cooldown:
            return None  # cooldown activo, ignorar
        self._ultimo_evento[tipo_evento] = ahora

        print(f"[{self.modulo}] Cam={self.camara_id} | EVENTO={tipo_evento} | dur={duracion_seg}s | datos={datos}")
        try:
            evento_id = insertar_evento(
                self.camara_id, self.modulo, tipo_evento, duracion_seg, datos
            )
            print(f"[{self.modulo}] Cam={self.camara_id} | Evento guardado en BD id={evento_id}")
        except Exception as e:
            print(f"[{self.modulo}] Cam={self.camara_id} | ERROR al insertar evento: {e}")
            return None

        # Push WebSocket al dashboard PHP en tiempo real
        self._push_ws(tipo_evento, duracion_seg, datos)

        # WhatsApp desactivado temporalmente
        # from services.redis_client import puede_alertar
        # from services.wasenger    import enviar_alerta
        # from services.database    import marcar_alerta_enviada
        # if duracion_seg >= self.umbral_seg and puede_alertar(self.camara_id, tipo_evento):
        #     min_fmt = self._fmt_tiempo(duracion_seg)
        #     msg = (
        #         f"⚠️ *Alicia IA — Alerta*\n"
        #         f"📷 Camara: {self.camara_id}\n"
        #         f"🔔 {tipo_evento.replace('_', ' ').title()}\n"
        #         f"⏱ Duracion: {min_fmt} min\n"
        #         f"📅 {time.strftime('%d/%m/%Y %H:%M:%S')}"
        #     )
        #     if enviar_alerta(msg):
        #         marcar_alerta_enviada(evento_id)

        return evento_id

    def _push_ws(self, tipo_evento: str, duracion_seg: int, datos: dict):
        """Envia el evento por WebSocket a todos los clientes del dashboard.
        Usa run_coroutine_threadsafe porque este metodo se llama desde un thread,
        no desde el loop principal de asyncio."""
        import asyncio
        from api.routes_ws import ws_manager, _main_loop

        payload = {
            "tipo":        tipo_evento,
            "camara_id":   self.camara_id,
            "duracion_seg": duracion_seg,
            "datos":       datos,
            "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        try:
            if _main_loop and _main_loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    ws_manager.broadcast(self.camara_id, payload),
                    _main_loop,
                )
        except Exception:
            pass
