# --- Importaciones Necesarias ---
from fastapi import FastAPI, Response, HTTPException, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
import os
from datetime import datetime, timedelta
import base64
from io import BytesIO
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from dotenv import load_dotenv
from fpdf import FPDF
import httpx
import json
import asyncio
from collections import deque, OrderedDict
import time
import re
import hashlib

from langchain_openai import ChatOpenAI
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun  # IMPORT CORREGIDO
from langchain.agents import create_react_agent, AgentExecutor, Tool
from langchain import hub

load_dotenv()

LM_STUDIO_URL = "-"  
OLLAMA_URL = "-"
OLLAMA_VISION_MODEL = "-"
HUGGINGFACE_API_URL = "-"
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

MAX_CONTEXT_TOKENS = 2000
MAX_HISTORY_MESSAGES = 6
QUICK_MODE_TOKENS = 0

app = FastAPI(title="Nube AI - OnixNube", version="15.0.0")

if os.path.isdir("static"):
    try:
        app.mount("/static", StaticFiles(directory="static"), name="static")
    except Exception as e:
        print(f"⚠️ No se pudo montar /static: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_origin_regex=r"https?://.*", 
    allow_credentials=False, 
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
)

class ResponseCache:
    
    def __init__(self, max_size: int = int(os.getenv("CACHE_SIZE", "100"))):
        self.cache: OrderedDict[str, str] = OrderedDict()
        self.max_size = max_size

    def get_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, text: str) -> Optional[str]:
        key = self.get_key(text)
        value = self.cache.get(key)
        if value is not None:
            self.cache.move_to_end(key)
        return value

    def set(self, text: str, response: str):
        key = self.get_key(text)
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = response
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

response_cache = ResponseCache()

class OptimizedRequestQueue:
    def __init__(self):
        self.queue = deque()
        self.quick_queue = deque()
        self.processing = False
        self.lock = asyncio.Lock()
        self.current_task = None
    
    async def add_request(self, request_func, is_quick=False):
        future = asyncio.Future()
        
        async with self.lock:
            if is_quick:
                self.quick_queue.appendleft((request_func, future))
            else:
                self.queue.append((request_func, future))
        
        if not self.processing:
            asyncio.create_task(self._process_queue())
        
        return await future
    
    async def _process_queue(self):
        async with self.lock:
            if self.processing:
                return
            self.processing = True
        
        try:
            while self.quick_queue or self.queue:
                if self.quick_queue:
                    request_func, future = self.quick_queue.popleft()
                    timeout = 15
                else:
                    request_func, future = self.queue.popleft()
                    timeout = 75
                
                self.current_task = asyncio.create_task(
                    self._execute_with_timeout(request_func, future, timeout)
                )
                await self.current_task
                await asyncio.sleep(0.01)
        finally:
            async with self.lock:
                self.processing = False
                self.current_task = None
    
    async def _execute_with_timeout(self, request_func, future, timeout):
        try:
            result = await asyncio.wait_for(request_func(), timeout=timeout)
            future.set_result(result)
        except asyncio.TimeoutError:
            future.set_exception(TimeoutError("Tiempo de espera agotado"))
        except Exception as e:
            future.set_exception(e)

request_queue = OptimizedRequestQueue()

class ImageRateLimiter:
    def __init__(self):
        self.limits = {}
        self.max_requests = 2
        self.window_hours = 3
    
    def check_limit(self, client_ip: str) -> tuple[bool, str]:
        current_time = datetime.now()
        
        if client_ip in self.limits:
            count, reset_time = self.limits[client_ip]
            if current_time < reset_time:
                if count >= self.max_requests:
                    remaining = (reset_time - current_time).total_seconds() / 60
                    return False, f"Límite alcanzado. Disponible en {int(remaining)} minutos."
                self.limits[client_ip] = (count + 1, reset_time)
            else:
                self.limits[client_ip] = (1, current_time + timedelta(hours=self.window_hours))
        else:
            self.limits[client_ip] = (1, current_time + timedelta(hours=self.window_hours))
        
        return True, "OK"

image_limiter = ImageRateLimiter()

class OptimizedWebSearch:
    def __init__(self):
        self.search = DuckDuckGoSearchRun()
        self.cache: Dict[str, Dict[str, Any]] = {}

    def search_blocking(self, query: str) -> str:
        if query in self.cache:
            age = time.time() - self.cache[query]['time']
            if age < 3600:
                return self.cache[query]['result']
        try:
            result = self.search.run(query)
            formatted = result[:1500]
            self.cache[query] = {'result': formatted, 'time': time.time()}
            return formatted
        except Exception as e:
            return f"Error búsqueda: {e}"

    async def search_async(self, query: str) -> str:
        if query in self.cache:
            age = time.time() - self.cache[query]['time']
            if age < 3600:
                return self.cache[query]['result']
        try:
            result = await asyncio.to_thread(self.search.run, query)
            if len(result) < 300:
                expanded_query = f"{query} información detallada actualizada 2024 2025"
                additional = await asyncio.to_thread(self.search.run, expanded_query)
                result = f"{result}\n\n{additional}"
            formatted = f"""
📊 **Búsqueda Web Actualizada**
Consulta: "{query}"

{result[:1500]}

_Fuente: Búsqueda web en tiempo real_
"""
            self.cache[query] = {'result': formatted, 'time': time.time()}
            return formatted
        except Exception as e:
            return f"No se pudo realizar la búsqueda: {e}"

web_search = OptimizedWebSearch()
agent_executor: Optional[AgentExecutor] = None

async def initialize_agent():
    global agent_executor
    try:
        llm_agent = ChatOpenAI(
            base_url=LM_STUDIO_URL,
            api_key="not-needed",
            model="local-model",
            temperature=0.7,
            max_tokens=512,
            request_timeout=30
        )
        tools = [
            Tool(
                name="web_search",
                func=web_search.search_blocking,
                description="Busca información básica (cache 1h)"
            )
        ]
        try:
            prompt = hub.pull("hwchase17/react-chat")
        except Exception:
            from langchain.prompts import PromptTemplate
            prompt = PromptTemplate(
                input_variables=["input", "chat_history", "agent_scratchpad"],
                template=(
                    "Eres Nube, asistente de IA.\n\n"
                    "Historial: {chat_history}\n\n"
                    "Pregunta: {input}\n\n"
                    "Razonamiento: {agent_scratchpad}\n"
                    "Responde claro y útil."
                )
            )
        agent = create_react_agent(llm_agent, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=1,
            max_execution_time=20
        )
        print("✅ Nube AI inicializado correctamente")
        return True
    except Exception as e:
        print(f"⚠️ Modo sin búsqueda web: {e}")
        agent_executor = None
        return False

class ArchivoAdjunto(BaseModel):
    type: str
    content: str
    name: str

class HistorialMensaje(BaseModel):
    role: str
    content: Any

class Pregunta(BaseModel):
    mensaje: str
    archivo: Optional[ArchivoAdjunto] = None
    instrucciones: Optional[str] = ""
    historial: Optional[List[HistorialMensaje]] = []
    modo: str = "chat"
    quick_mode: bool = False
    client_ip: Optional[str] = "127.0.0.1"

def estimate_tokens(text: str) -> int:
    return len(text) // 4

def smart_truncate_context(messages: List[HistorialMensaje], max_tokens: int, quick_mode: bool = False) -> List[HistorialMensaje]:
    if quick_mode or not messages:
        return []
    
    total_tokens = 0
    truncated = []
    
    for msg in reversed(messages[-MAX_HISTORY_MESSAGES:]):
        content = msg.content if isinstance(msg.content, str) else str(msg.content.get('text', ''))
        
        if len(content) > 300:
            content = content[:300] + "..."
        
        msg_tokens = estimate_tokens(content)
        
        if total_tokens + msg_tokens > max_tokens:
            break
            
        msg_copy = HistorialMensaje(role=msg.role, content=content)
        truncated.insert(0, msg_copy)
        total_tokens += msg_tokens
    
    return truncated


@app.on_event("startup")
async def startup_event():
    await initialize_agent()
    print("🚀 Servidor Nube optimizado para máxima velocidad")

@app.get("/")
async def read_index():
    if os.path.exists('index.html'):
        return FileResponse('index.html')
    else:
        return {"message": "Nube AI Server Running"}

@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "ai_model": "Nube AI",
        "version": app.version,
        "performance": "optimized",
        "capabilities": {
            "text": "✅ Activo",
            "vision": "✅ Activo (Rápido)",
            "search": "✅ Activo",
            "image_gen": "✅ Activo" if HUGGINGFACE_TOKEN else "⚠️ No configurado",
            "quick_mode": "✅ Ultra-rápido"
        }
    }

@app.get("/config")
async def get_config():
    """Endpoint simple para exponer configuración pública utilizable por frontends estáticos."""
    return {
        "version": app.version,
        "endpoints": {
            "chat": "/chat",
            "health": "/health",
            "pdf": "/generate_pdf"
        }
    }

@app.get("/manifest.json")
async def pwa_manifest():
    """Manifest básico para evitar 404 y permitir instalación PWA mínima."""
    manifest_path = os.path.join(os.getcwd(), "manifest.json")
    if os.path.exists(manifest_path):
        return FileResponse(manifest_path, media_type="application/manifest+json")
    return {
        "name": "Nube AI",
        "short_name": "NubeAI",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#0a0a0a",
        "theme_color": "#0a0a0a",
        "icons": [
            {"src": "/static/AVC.png", "sizes": "512x512", "type": "image/png"}
        ]
    }

@app.get("/AVC.png")
async def get_root_icon():
    """Sirve el ícono AVC.png desde raíz o desde /static para evitar 404 en navegadores."""
    try_paths = [os.path.join(os.getcwd(), "AVC.png"), os.path.join(os.getcwd(), "static", "AVC.png")]
    for p in try_paths:
        if os.path.exists(p):
            return FileResponse(p, media_type="image/png")
    raise HTTPException(status_code=404, detail="AVC.png no encontrado")

@app.post("/chat")
async def procesar_chat_router(request: Request, pregunta: Pregunta):
    try:
        client_ip = request.client.host if request.client else "127.0.0.1"
        pregunta.client_ip = client_ip
    except:
        pass
    
    async def process():
        try:
            if pregunta.quick_mode and pregunta.archivo is not None:
                return {"error": "El Modo Rápido no admite adjuntos", "code": "quick_mode_no_attachments"}
            if pregunta.quick_mode and not pregunta.archivo:
                cached = response_cache.get(pregunta.mensaje)
                if cached:
                    return {"contenido": {"type": "text", "text": cached, "cached": True}}

            if pregunta.modo == "generate_image":
                return await generar_imagen_optimizada(pregunta)
            if pregunta.archivo and pregunta.archivo.type == 'image':
                return await analizar_imagen_rapida(pregunta)
            if pregunta.archivo and pregunta.archivo.type == 'document':
                return await procesar_texto_ultra_rapido(pregunta)
            return await procesar_texto_ultra_rapido(pregunta)

        except (asyncio.TimeoutError, TimeoutError):
            return {"error": "Tiempo excedido. Intenta nuevamente.", "code": "timeout"}
        except Exception as e:
            print(f"[ERROR /chat] {e}")
            return {"error": "Error interno procesando la solicitud", "code": "internal"}
    
    return await request_queue.add_request(process, is_quick=pregunta.quick_mode)

@app.post("/generate_pdf")
async def generar_pdf(data: Dict):
    try:
        html_content = data.get("html", "")
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=11)
        
        pdf.cell(0, 10, txt="Nube AI - Conversacion", ln=True, align='C')
        pdf.ln(5)
        
        text = re.sub('<[^<]+?>', '', html_content)
        text = text.replace('&nbsp;', ' ').replace('&amp;', '&')
        
        for line in text.split('\n')[:100]:
            if line.strip():
                try:
                    safe_line = line.encode('latin-1', 'ignore').decode('latin-1')
                    pdf.multi_cell(0, 8, txt=safe_line.strip()[:100], align='L')
                except:
                    pass
        
        pdf_output = pdf.output(dest='S').encode('latin-1', 'ignore')
        
        return Response(
            content=pdf_output,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=nube_chat.pdf"}
        )
        
    except Exception as e:
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 10, txt="Nube AI - PDF", ln=True, align='C')
            pdf.ln(10)
            msg = f"No se pudo generar el PDF completo. Detalle: {str(e)[:120]}"
            safe_msg = msg.encode('latin-1', 'ignore').decode('latin-1')
            pdf.multi_cell(0, 8, txt=safe_msg, align='L')
            out = pdf.output(dest='S').encode('latin-1', 'ignore')
            return Response(
                content=out,
                media_type="application/pdf",
                headers={"Content-Disposition": "attachment; filename=nube_chat_error.pdf"}
            )
        except:
            raise HTTPException(status_code=500, detail="No se pudo generar el PDF")


async def procesar_texto_ultra_rapido(pregunta: Pregunta):
    mensaje_completo = pregunta.mensaje
    respuesta = ""

    if pregunta.quick_mode:
        messages = [
            {"role": "system", "content": "Eres Nube, responde directamente."},
            {"role": "user", "content": mensaje_completo}
        ]
    else:
        if pregunta.archivo and pregunta.archivo.type == 'document':
            doc_content = pregunta.archivo.content[:1000]
            mensaje_completo = f"Documento: {doc_content}\n\nPregunta: {pregunta.mensaje}"
        
        if pregunta.instrucciones and not pregunta.quick_mode:
            mensaje_completo = f"{pregunta.instrucciones}\n\n{mensaje_completo}"
        
        messages = [
            {"role": "system", "content": "Eres Nube, asistente IA rápido y preciso."}
        ]
        
        max_tokens = 0 if pregunta.quick_mode else MAX_CONTEXT_TOKENS
        truncated_history = smart_truncate_context(
            pregunta.historial, 
            max_tokens, 
            pregunta.quick_mode
        )
        
        for msg in truncated_history:
            content = msg.content if isinstance(msg.content, str) else msg.content.get('text', '')
            messages.append({
                "role": msg.role if msg.role in ["user", "assistant"] else "assistant",
                "content": content[:200]
            })
        
        messages.append({"role": "user", "content": mensaje_completo})
    
    try:
        if pregunta.modo == "search":
            result = await web_search.search_async(mensaje_completo)
            if pregunta.quick_mode:
                return {"contenido": {"type": "text", "text": result.strip() or "Sin resultados"}}
            try:
                async with httpx.AsyncClient(timeout=35.0) as client:
                    search_messages = [
                        {"role": "system", "content": "Resume claramente la información y cita las fuentes si están en el texto."},
                        {"role": "user", "content": f"Pregunta: {pregunta.mensaje}\n\nResultados de la web:\n{result}"}
                    ]
                    r = await client.post(
                        f"{LM_STUDIO_URL}/chat/completions",
                        json={
                            "model": "local-model",
                            "messages": search_messages,
                            "temperature": 0.5,
                            "max_tokens": 600,
                            "stream": False
                        }
                    )
                if r.status_code == 200:
                    data = r.json()
                    respuesta = data.get("choices", [{}])[0].get("message", {}).get("content", "") or result
                else:
                    respuesta = result
            except Exception:
                respuesta = result
        else:
            http_timeout = 12.0 if pregunta.quick_mode else 40.0
            async with httpx.AsyncClient(timeout=http_timeout) as client:
                r = await client.post(
                    f"{LM_STUDIO_URL}/chat/completions",
                    json={
                        "model": "local-model",
                        "messages": messages,
                        "temperature": 0.7,
                        "max_tokens": 300 if pregunta.quick_mode else 1200,
                        "stream": False
                    }
                )
                if r.status_code == 200:
                    data = r.json()
                    respuesta = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if pregunta.quick_mode and respuesta:
                        response_cache.set(pregunta.mensaje, respuesta)
                else:
                    respuesta = "Error temporal. Por favor intenta de nuevo."
        if not respuesta:
            respuesta = "Sin respuesta disponible."
        return {"contenido": {"type": "text", "text": respuesta.strip()}}
    except Exception as e:
        print(f"[ERROR procesar_texto_ultra_rapido] {e}")
        return {"error": "Error procesando. Intenta con el modo rápido."}

async def analizar_imagen_rapida(pregunta: Pregunta):
    """Pipeline visión -> LM Studio: describe con Ollama (bakllava) y formula respuesta final con LM Studio."""
    user_message = {
        "role": "user",
        "content": pregunta.mensaje or "Describe esta imagen brevemente."
    }

    try:
        base64_content = pregunta.archivo.content.split(',', 1)[1]
        user_message["images"] = [base64_content]
    except Exception as e:
        return {"error": f"Formato de imagen inválido: {e}", "code": "invalid_image"}

    vision_payload = {
        "model": OLLAMA_VISION_MODEL,
        "messages": [user_message],
        "stream": False,
        "options": {
            "temperature": 0.4,
            "num_predict": 300,
            "top_k": 10,
            "top_p": 0.9
        }
    }

    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            vres = await client.post(OLLAMA_URL, json=vision_payload)
            if vres.status_code != 200:
                detail = vres.text[:200]
                return {"error": f"Visión falló ({vres.status_code}): {detail}", "code": "vision_http"}

            vjson = vres.json()
            vision_text = vjson.get("message", {}).get("content") or "No se pudo analizar la imagen"
            vision_text = vision_text.strip()

            lm_messages = [
                {"role": "system", "content": "Eres un asistente que redacta descripciones claras y útiles."},
                {"role": "user", "content": (
                    f"Usuario: {pregunta.mensaje or 'Describe la imagen.'}\n\n"
                    f"Descripción visual (modelo de visión): {vision_text}\n\n"
                    f"Tarea: Redacta una descripción final clara y completa para el usuario basándote en la descripción visual."
                )}
            ]

            lres = await client.post(
                f"{LM_STUDIO_URL}/chat/completions",
                json={
                    "model": "local-model",
                    "messages": lm_messages,
                    "temperature": 0.6,
                    "max_tokens": 500,
                    "stream": False
                }
            )

            if lres.status_code == 200:
                ljson = lres.json()
                final_text = ljson.get("choices", [{}])[0].get("message", {}).get("content")
                final_text = (final_text or vision_text or "No hay descripción disponible").strip()
                return {"contenido": {"type": "text", "text": final_text}}
            else:
                return {"contenido": {"type": "text", "text": vision_text or "No hay descripción disponible"}}

    except Exception as e:
        return {"error": f"Error analizando imagen: {str(e)}", "code": "vision_error"}

async def generar_imagen_optimizada(pregunta: Pregunta):
    allowed, message = image_limiter.check_limit(pregunta.client_ip)
    if not allowed:
        return {"error": message}
    
    if not HUGGINGFACE_TOKEN:
        return {"error": "Generación de imágenes no configurada"}
    
    headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}
    enhanced = f"{pregunta.mensaje}, high quality, detailed"
    payload = {"inputs": enhanced[:200]}
    
    async with httpx.AsyncClient(timeout=45.0) as client:
        try:
            response = await client.post(
                HUGGINGFACE_API_URL,
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                image_bytes = response.content
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                image_data_url = f"data:image/png;base64,{image_base64}"
                
                return {"contenido": {"type": "image", "text": image_data_url}}
            else:
                return {"error": "No se pudo generar la imagen", "code": "image_generation_failed"}
                
        except Exception as e:
            return {"error": f"Error: {str(e)}", "code": "image_exception"}

if __name__ == "__main__":
    import uvicorn
    
    try:
        from pyngrok import ngrok
        public_url = ngrok.connect(8000)
        print(f"🌐 Ngrok: {public_url}")
    except:
        print("ℹ️ Ngrok no disponible")
    
    print("=" * 50)
    print("🚀 NUBE AI - Sistema Optimizado")
    print("⚡ Modo Rápido disponible")
    print("🧠 Modelo: Nube (OnixNube)")
    print("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False, log_level="warning")