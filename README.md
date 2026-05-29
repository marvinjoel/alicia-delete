# Alicia IA — Servicio de Vision Artificial

Servicio Python que procesa streams de video con YOLO, detecta eventos en restaurantes y notifica via WhatsApp.

## Stack

- **Python 3.10+**
- **FastAPI** — API REST + WebSocket
- **Ultralytics YOLO** — deteccion de objetos (yolo11 / yolov8)
- **OpenCV** — captura y procesamiento de frames
- **MySQL** — misma BD que el panel PHP (`alicia_ia`)
- **Redis** — estado de camaras y cooldown de alertas
- **Wasenger** — notificaciones WhatsApp

## Flujo

```
[Camara RTSP/HTTP/USB]
       | cv2.VideoCapture(url)
       v
[Frame BGR por OpenCV]
       | modelo(frame)
       v
[YOLO detecta objetos: cajas, clases, confianza]
       | procesador.procesar()
       v
[Logica: cronometrar, contar, alertar]
       | cv2.imencode('.jpg')
       v
[Frame JPEG en RAM]
       |
       |-- GET /camaras/{id}/stream  →  MJPEG  →  <img> en PHP
       |-- WS  /ws/{id}              →  push JSON evento al dashboard
       └-- Wasenger API              →  WhatsApp al numero configurado
```

## Instalacion

### Ambiente local (Laragon / Windows)

```bash
# 1. Crear entorno virtual (solo una vez)
python -m venv venv

# 2. Activar el entorno virtual
venv\Scripts\activate
# El prompt cambia a: (venv) D:\...\alicia_ia>

# 3. Instalar dependencias (dentro del venv)
pip install -r requirements.txt

# 4. Configurar variables de entorno
copy config.env.example config.env
# Abrir config.env y completar: DB_HOST, WASENGER_API_KEY, WASENGER_NUMERO_DESTINO

# 5. Crear la base de datos en MySQL
# Abrir HeidiSQL o phpMyAdmin y ejecutar:
#   documentacion/bd.sql  (desde el proyecto alicia/)

# 6. Iniciar Redis
# Descargar Redis para Windows: https://github.com/microsoftarchive/redis/releases

# 7. Arrancar el servicio (modo desarrollo con recarga automatica)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

> Cada vez que abras una terminal nueva debes activar el venv primero:
> `venv\Scripts\activate`

**Script rapido** — crear `arrancar.bat` en la raiz del proyecto:
```bat
@echo off
cd /d D:\Instalaciones\laragon\www\alicia_ia
call venv\Scripts\activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
pause
```

---

### Produccion (servidor Linux / VPS)

En produccion se usa `venv` para aislar dependencias y `systemd` para que
el servicio arranque automaticamente y se recupere si falla.

```bash
# 1. Clonar / subir el proyecto
cd /var/www/alicia_ia

# 2. Crear entorno virtual e instalar dependencias
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Configurar variables de entorno
cp config.env.example config.env
nano config.env   # completar credenciales reales

# 4. Probar que arranca correctamente
uvicorn main:app --host 127.0.0.1 --port 8000
# Ctrl+C para detener, luego registrar el servicio
```

**Registrar como servicio con systemd:**

Crear el archivo `/etc/systemd/system/alicia-ia.service`:
```ini
[Unit]
Description=Alicia IA — Vision Artificial
After=network.target mysql.service redis.service

[Service]
User=www-data
WorkingDirectory=/var/www/alicia_ia
ExecStart=/var/www/alicia_ia/venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000 --workers 1
Restart=always
RestartSec=5
EnvironmentFile=/var/www/alicia_ia/config.env

[Install]
WantedBy=multi-user.target
```

```bash
# Activar e iniciar el servicio
systemctl daemon-reload
systemctl enable alicia-ia
systemctl start alicia-ia

# Verificar estado
systemctl status alicia-ia

# Ver logs en tiempo real
journalctl -u alicia-ia -f
```

> **Nota:** En produccion usar `--workers 1` porque los workers de camara
> viven en memoria RAM. Con multiples workers de uvicorn cada proceso
> tendria su propio diccionario de workers y el stream fallaria.

## Endpoints

| Metodo | URL | Descripcion |
|--------|-----|-------------|
| GET | `/` | Health check |
| POST | `/camaras/{id}/iniciar` | Inicia worker IA para la camara |
| POST | `/camaras/{id}/detener` | Detiene el worker |
| GET | `/camaras/{id}/stream` | Stream MJPEG anotado con overlays |
| GET | `/eventos` | Listado de eventos recientes |
| WS | `/ws/{camara_id}` | Push de eventos en tiempo real |

## Modelos YOLO

Los modelos se descargan automaticamente en el primer uso.
Colocarlos en la carpeta `models/` para evitar re-descarga.

| Archivo | Velocidad | Cuando usarlo |
|---------|-----------|---------------|
| `yolo11n.pt` | Muy rapido | CPU, muchas camaras |
| `yolo11m.pt` | Medio | Recomendado para produccion |
| `yolov8m.pt` | Medio | Alternativa estable |

## Modulos de Analitica

| Modulo | Clase detectada | Alerta |
|--------|----------------|--------|
| `dirty_table_detector` | Platos, vasos, cubiertos en mesa | Mesa sucia X minutos |
| `table_occupancy` | Personas sentadas | % ocupacion del salon |
| `customer_wait_time` | Persona sentada sin mesero cerca | Cliente esperando X min |
| `staff_tracker` | Persona con uniforme por zona | Mesero inactivo X min |
| `people_counter` | Todas las personas | Aforo excedido |
| `queue_detector` | Personas de pie en entrada | Cola de N personas |

## Estructura del Proyecto

```
alicia_ia/
├── main.py                  <- FastAPI app, registra routers
├── config.env               <- Variables de entorno (no subir a git)
├── config.env.example       <- Plantilla de configuracion
├── requirements.txt
├── models/                  <- Archivos .pt de YOLO
├── processors/
│   ├── base_processor.py    <- Clase base con logica comun
│   ├── dirty_tables.py
│   ├── table_occupancy.py
│   ├── customer_wait_time.py
│   ├── staff_tracker.py
│   ├── people_counter.py
│   └── queue_detector.py
├── api/
│   ├── routes_camaras.py    <- REST iniciar/detener
│   ├── routes_eventos.py    <- REST historial
│   ├── routes_stream.py     <- MJPEG stream
│   └── routes_ws.py         <- WebSocket manager
├── services/
│   ├── database.py          <- Conexion MySQL
│   ├── wasenger.py          <- WhatsApp API
│   └── redis_client.py      <- Cache y estado
└── workers/
    └── camera_worker.py     <- Thread por camara activa
```




Resumen de la integración FastVLM
Archivos creados (backend Python):
Archivo	Descripción
services/vlm_service.py	Singleton que carga FastVLM, ejecuta inferencia thread-safe, loop por cámara, broadcast WebSocket
api/routes_vlm.py	Endpoints: POST /vlm/{id}/iniciar, POST /vlm/{id}/detener, GET /vlm/estado, GET /vlm/{id}/historial
Archivos modificados (backend):
Archivo	Cambio
workers/camera_worker.py	Agrega self.frame_raw — frame limpio (sin anotar) que VLM usa para describir
services/database.py	Agrega insertar_vision_log() y get_vision_log()
main.py	Registra router_vlm
config.env	Agrega VLM_MODELO_DEFAULT y VLM_INTERVALO_SEG
requirements.txt	Agrega torch, transformers, accelerate, Pillow
Archivo modificado (frontend):
Archivo	Cambio
detalle_camara.tpl.php	Reemplaza VLM navegador (Transformers.js) con FastVLM servidor: selector de modelo, WebSocket para recibir descripciones
Flujo de funcionamiento:

1. Usuario selecciona modelo (0.5B/1.5B/7B) y click "Iniciar Vision IA"
2. Frontend → POST /vlm/{cam_id}/iniciar?modelo=FastVLM-0.5B&intervalo=10
3. Backend carga modelo (primera vez descarga de HuggingFace)
4. Loop cada N segundos: toma frame_raw del worker → PIL → FastVLM → descripción español
5. Descripción → INSERT vision_ia_log + broadcast WebSocket {"tipo": "vision_ia"}
6. Frontend recibe via WebSocket → muestra en log de Vision IA
Base de datos:
La tabla vision_ia_log se crea automáticamente al iniciar el servicio.

Para instalar las dependencias:

cd d:\Instalaciones\laragon\www\alicia_ia
pip install torch transformers accelerate Pillow
# Para GPU NVIDIA (mucho más rápido):
# pip install torch --index-url https://download.pytorch.org/whl/cu121
Notas:
FastVLM-0.5B es el recomendado para desarrollo (~1GB RAM, rápido en CPU)
FastVLM-7B necesita GPU con ~14GB VRAM
El modelo se descarga una sola vez (se cachea en ~/.cache/huggingface/)
YOLO y VLM corren en threads separados, no se bloquean entre sí

modelo se descarag en 
C:\Users\admin\.cache\huggingface\hub\models--apple--FastVLM-0.5B\


# cache para el modelo

No es recomendable para producción. En el server:

El caché quedaría en /root/.cache/ o /home/usuario/.cache/ — difícil de controlar y respaldar.
Si usas Docker, se pierde al recrear el contenedor (hay que re-descargar ~1GB).
Recomendado: define una ruta fija en config.env:


HF_HOME=/opt/modelos_ia
(En Windows local puedes usar: HF_HOME=d:\modelos_ia)

Ventajas:

Ruta predecible, fácil de respaldar
Puedes pre-descargar los modelos antes del deploy
En Docker, se monta como volumen persistente

Local: caché queda en C:\Users\admin\.cache\huggingface\ (intacto, FastVLM ya descargado)
Producción: solo descomentas HF_HOME en config.env y pones una ruta absoluta del servidor



llava-v1.6-mistral-7b o Qwen2-VL (más pesados)