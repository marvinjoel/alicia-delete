import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, List

router = APIRouter()


class ConnectionManager:
    """Gestiona las conexiones WebSocket activas por camara."""

    def __init__(self):
        # camara_id -> lista de WebSocket conectados
        self.conexiones: Dict[int, List[WebSocket]] = {}

    async def conectar(self, ws: WebSocket, camara_id: int):
        await ws.accept()
        self.conexiones.setdefault(camara_id, []).append(ws)
        print(f"[WS] Cliente conectado a camara {camara_id}. Total: {len(self.conexiones[camara_id])}")

    def desconectar(self, ws: WebSocket, camara_id: int):
        if camara_id in self.conexiones:
            self.conexiones[camara_id].remove(ws)

    async def broadcast(self, camara_id: int, data: dict):
        """Envia un evento JSON a todos los clientes escuchando esta camara."""
        clientes = self.conexiones.get(camara_id, [])
        caidos = []
        for ws in clientes:
            try:
                await ws.send_text(json.dumps(data, default=str))
            except Exception:
                caidos.append(ws)
        for ws in caidos:
            self.desconectar(ws, camara_id)


# Instancia global accesible desde los workers
ws_manager = ConnectionManager()

# Loop principal de asyncio — se asigna al arrancar FastAPI
_main_loop = None


@router.websocket("/ws/{camara_id}")
async def websocket_endpoint(ws: WebSocket, camara_id: int):
    from workers.camera_worker import workers_activos
    
    await ws_manager.conectar(ws, camara_id)
    try:
        while True:
            # 1. El backend ahora se queda escuchando comandos del frontend
            data = await ws.receive_text()
            
            try:
                msg = json.loads(data)
                
                # 2. Si el frontend envía la orden de cambiar clases:
                # Payload esperado del front: {"action": "update_classes", "classes": [0, 39]}
                if msg.get("action") == "update_classes":
                    nuevas_clases = msg.get("classes")
                    
                    worker = workers_activos.get(camara_id)
                    if worker and hasattr(worker, 'modelo'):
                        # YOLO actualiza el filtro en tiempo real en la GPU
                        worker.modelo.classes = nuevas_clases if nuevas_clases else None
                        print(f"[WS] Cam {camara_id} | Filtro de clases actualizado: {nuevas_clases}")
                        
            except json.JSONDecodeError:
                print(f"[WS] Error decodificando mensaje de control: {data}")
                
    except WebSocketDisconnect:
        ws_manager.desconectar(ws, camara_id)
        print(f"[WS] Cliente desconectado de camara {camara_id}")
