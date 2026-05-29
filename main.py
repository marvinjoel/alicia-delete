from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv("config.env")

from api.routes_camaras import router as router_camaras
from api.routes_eventos  import router as router_eventos
from api.routes_stream   import router as router_stream
from api.routes_ws       import router as router_ws
from api.routes_webcam   import router as router_webcam
from api.routes_legos    import router as router_legos
from api.routes_vlm      import router as router_vlm

app = FastAPI(
    title="Alicia IA",
    description="Servicio de vision artificial para restaurantes",
    version="1.0.0",
    root_path="/alicia-back"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router_camaras, tags=["Camaras"])
app.include_router(router_eventos,  tags=["Eventos"])
app.include_router(router_stream,   tags=["Stream"])
app.include_router(router_ws,       tags=["WebSocket"])
app.include_router(router_webcam,   tags=["Webcam"])
app.include_router(router_legos,    tags=["Legos"])
app.include_router(router_vlm)


@app.on_event("startup")
async def _capturar_loop():
    """Guarda el loop principal para que los threads puedan programar coroutines."""
    import asyncio
    import api.routes_ws as ws_mod
    ws_mod._main_loop = asyncio.get_event_loop()


@app.get("/")
def health():
    from workers.camera_worker import workers_activos
    return {
        "status": "ok",
        "servicio": "Alicia IA",
        "camaras_activas": list(workers_activos.keys()),
    }
