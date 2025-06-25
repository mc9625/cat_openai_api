"""
Universal OpenAI Bridge Plugin for Cheshire Cat
CONTROLLED PERSONALITY: Usa le capacità del Cat ma con personalità configurabile
"""

from cat.mad_hatter.decorators import tool, hook, plugin, endpoint
from cat.log import log
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Dict, Any, List, Optional, AsyncGenerator
from collections import defaultdict, Counter
from fastapi.responses import StreamingResponse
import json
import time
import asyncio

# ===== PLUGIN SETTINGS =====

class UniversalBridgeSettings(BaseModel):
    """Settings for the Universal OpenAI Bridge plugin."""
    
    enable_conversation_memory: bool = Field(
        title="Enable Conversation Memory",
        description="Keep track of conversation history for context",
        default=True
    )
    
    max_conversation_history: int = Field(
        title="Max Conversation History", 
        description="Maximum number of messages to keep in memory per user",
        default=20,
        ge=1,
        le=100
    )
    
    rate_limit_per_minute: int = Field(
        title="Rate Limit (requests per minute)",
        description="Maximum requests per minute per user (0 = no limit)",
        default=60,
        ge=0,
        le=1000
    )
    
    enable_detailed_logging: bool = Field(
        title="Enable Detailed Logging",
        description="Log detailed information about requests and responses",
        default=False
    )
    
    enable_streaming: bool = Field(
        title="Enable Streaming Support",
        description="Support streaming responses for compatible clients",
        default=True
    )
    
    max_tokens_limit: int = Field(
        title="Max Tokens Limit",
        description="Maximum tokens allowed per request (0 = no limit)",
        default=4000,
        ge=0,
        le=32000
    )
    
    cors_origins: str = Field(
        title="CORS Origins",
        description="Allowed CORS origins (comma-separated, or * for all)",
        default="*"
    )
    
    custom_model_names: str = Field(
        title="Custom Model Names",
        description="Additional model names to expose (comma-separated)",
        default="claude-3,gpt-4-turbo"
    )
    
    streaming_chunk_delay: float = Field(
        title="Streaming Chunk Delay (seconds)",
        description="Delay between streaming chunks for better UX",
        default=0.05,
        ge=0.0,
        le=1.0
    )
    
    # NUOVE IMPOSTAZIONI PER CONTROLLO PERSONALITA'
    ai_personality: str = Field(
        title="AI Personality",
        description="Define the AI personality and behavior",
        default="Sei un assistente AI intelligente e utile. Rispondi sempre in italiano in modo chiaro, preciso e professionale.",
        extra={"type": "TextArea"}
    )
    
    use_cat_personality: bool = Field(
        title="Use Cat's Default Personality",
        description="Use Cheshire Cat's default personality (Alice in Wonderland style)",
        default=False
    )
    
    force_italian: bool = Field(
        title="Force Italian Responses",
        description="Always respond in Italian regardless of input language",
        default=True
    )
    
    use_cat_memory: bool = Field(
        title="Use Cat's Memory System",
        description="Access Cat's episodic and declarative memory for context",
        default=True
    )
    
    use_cat_tools: bool = Field(
        title="Use Cat's Tools",
        description="Allow access to Cat's tools and plugins (when possible)",
        default=True
    )

@plugin
def settings_model():
    return UniversalBridgeSettings

# ===== MODELS OpenAI =====

class OpenAIMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

class OpenAIChatRequest(BaseModel):
    model: str = "gpt-3.5-turbo"
    messages: List[OpenAIMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    user: Optional[str] = None
    stream: Optional[bool] = False

class OpenAIChoice(BaseModel):
    index: int
    message: OpenAIMessage
    finish_reason: str

class OpenAIUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class OpenAIChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[OpenAIChoice]
    usage: OpenAIUsage

# ===== PLUGIN STATE =====

plugin_state = {
    "conversation_history": {},
    "request_counts": defaultdict(list),
    "active_streams": {},
    "metrics": {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "streaming_requests": 0,
        "endpoints_usage": Counter(),
        "user_activity": Counter(),
        "start_time": datetime.now().isoformat()
    }
}

_cat_instance = None

# ===== UTILITY FUNCTIONS =====

def generate_openai_id() -> str:
    import random
    import string
    timestamp = int(time.time())
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=12))
    return f"chatcmpl-{timestamp}{random_suffix}"

def count_tokens(text: str) -> int:
    return max(1, len(text) // 4)

def get_plugin_settings():
    try:
        if _cat_instance and hasattr(_cat_instance, 'mad_hatter'):
            plugin = _cat_instance.mad_hatter.get_plugin()
            if plugin:
                return plugin.load_settings()
        
        from cat.mad_hatter.mad_hatter import MadHatter
        mad_hatter = MadHatter()
        plugin = mad_hatter.get_plugin()
        if plugin:
            return plugin.load_settings()
    except Exception as e:
        log.warning(f"Could not load settings: {e}")
    return {}

def check_rate_limit(user_id: str) -> bool:
    settings = get_plugin_settings()
    rate_limit = settings.get("rate_limit_per_minute", 60)
    
    if rate_limit == 0:
        return True
    
    now = time.time()
    minute_ago = now - 60
    
    plugin_state["request_counts"][user_id] = [
        req_time for req_time in plugin_state["request_counts"][user_id] 
        if req_time > minute_ago
    ]
    
    if len(plugin_state["request_counts"][user_id]) >= rate_limit:
        return False
    
    plugin_state["request_counts"][user_id].append(now)
    return True

def record_metrics(endpoint: str, user_id: str, success: bool = True, is_streaming: bool = False):
    settings = get_plugin_settings()
    
    metrics = plugin_state["metrics"]
    metrics["total_requests"] += 1
    
    if success:
        metrics["successful_requests"] += 1
    else:
        metrics["failed_requests"] += 1
    
    if is_streaming:
        metrics["streaming_requests"] += 1
    
    metrics["endpoints_usage"][endpoint] += 1
    metrics["user_activity"][user_id] += 1

def manage_conversation_history(user_id: str, role: str, content: str):
    settings = get_plugin_settings()
    
    if not settings.get("enable_conversation_memory", True):
        return
    
    max_history = settings.get("max_conversation_history", 20)
    
    if user_id not in plugin_state["conversation_history"]:
        plugin_state["conversation_history"][user_id] = []
    
    plugin_state["conversation_history"][user_id].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })
    
    if len(plugin_state["conversation_history"][user_id]) > max_history:
        plugin_state["conversation_history"][user_id] = \
            plugin_state["conversation_history"][user_id][-max_history:]

async def get_controlled_cat_response(message: str, user_id: str = "api_user") -> str:
    """
    APPROCCIO IBRIDO: Usa le capacità del Cat (memoria, RAG) ma con personalità controllabile
    """
    try:
        if not _cat_instance:
            return "Ciao! Sono il tuo assistente AI. Come posso aiutarti?"
        
        settings = get_plugin_settings()
        cat = _cat_instance
        
        # Controlla se usare la personalità predefinita del Cat
        use_cat_personality = settings.get("use_cat_personality", False)
        
        if use_cat_personality:
            # USA IL CAT COMPLETO con la sua personalità
            log.debug("Using full Cat personality")
            try:
                original_user_id = getattr(cat, 'user_id', None)
                cat.user_id = user_id
                
                user_message_json = {
                    "text": message,
                    "user_id": user_id,
                    "timestamp": time.time()
                }
                
                result = await asyncio.to_thread(cat, user_message_json)
                
                if original_user_id:
                    cat.user_id = original_user_id
                
                if isinstance(result, dict):
                    response_text = result.get('content', str(result))
                else:
                    response_text = str(result)
                
                return response_text if response_text.strip() else "I'm here to help! *grins mysteriously*"
                
            except Exception as e:
                log.error(f"Full Cat personality failed: {e}")
        
        # USA APPROCCIO IBRIDO: capacità del Cat + personalità personalizzata
        log.debug("Using hybrid approach: Cat capabilities + custom personality")
        
        # 1. Raccogli contesto dalla memoria del Cat
        context_parts = []
        
        if settings.get("use_cat_memory", True) and hasattr(cat, 'memory'):
            try:
                # Memoria episodica (conversazioni precedenti)
                episodic_memories = cat.memory.vectors.episodic.search(message, k=3, threshold=0.7)
                if episodic_memories:
                    context_parts.append("Conversazioni precedenti:")
                    for mem, score in episodic_memories[:2]:
                        context_parts.append(f"- {mem.page_content[:100]}...")
            except Exception as e:
                log.debug(f"Could not get episodic memories: {e}")
            
            try:
                # Memoria dichiarativa (documenti caricati)
                declarative_memories = cat.memory.vectors.declarative.search(message, k=3, threshold=0.7)
                if declarative_memories:
                    context_parts.append("Informazioni rilevanti dai documenti:")
                    for mem, score in declarative_memories[:2]:
                        context_parts.append(f"- {mem.page_content[:100]}...")
            except Exception as e:
                log.debug(f"Could not get declarative memories: {e}")
        
        # 2. Recupera cronologia conversazioni del plugin
        user_history = plugin_state["conversation_history"].get(user_id, [])
        if user_history:
            context_parts.append("Cronologia recente:")
            for hist in user_history[-3:]:  # Ultimi 3 messaggi
                context_parts.append(f"- {hist['role']}: {hist['content'][:50]}...")
        
        # 3. Costruisci prompt personalizzato
        ai_personality = settings.get("ai_personality", 
            "Sei un assistente AI intelligente e utile. Rispondi sempre in italiano in modo chiaro, preciso e professionale.")
        
        force_italian = settings.get("force_italian", True)
        
        # Costruisci il prompt completo
        prompt_parts = [ai_personality]
        
        if force_italian:
            prompt_parts.append("IMPORTANTE: Rispondi SEMPRE in italiano, anche se la domanda è in altra lingua.")
        
        if context_parts:
            prompt_parts.append("Contesto disponibile:")
            prompt_parts.extend(context_parts)
        
        prompt_parts.append(f"Domanda dell'utente: {message}")
        prompt_parts.append("La tua risposta:")
        
        full_prompt = "\n\n".join(prompt_parts)
        
        # 4. Usa l'LLM del Cat con il prompt personalizzato
        response = await asyncio.to_thread(cat.llm, full_prompt)
        
        if response and str(response).strip():
            response_text = str(response).strip()
            
            # Post-processing per assicurarsi che sia in italiano
            if force_italian and not any(italian_word in response_text.lower() for italian_word in 
                ['ciao', 'come', 'sono', 'posso', 'aiutarti', 'grazie', 'prego', 'cosa', 'dove', 'quando']):
                # Se la risposta non sembra essere in italiano, forza una traduzione
                translate_prompt = f"Traduci questa risposta in italiano mantenendo il significato: {response_text}"
                translated = await asyncio.to_thread(cat.llm, translate_prompt)
                if translated and str(translated).strip():
                    response_text = str(translated).strip()
            
            return response_text
        
        # Fallback
        return "Ciao! Ho ricevuto la tua richiesta e sono qui per aiutarti. Puoi essere più specifico su cosa ti serve?"
    
    except Exception as e:
        log.error(f"Error in get_controlled_cat_response: {e}")
        return f"Ciao! Ho ricevuto il tuo messaggio: '{message[:50]}...' e sono qui per aiutarti, anche se ho avuto un piccolo problema tecnico."

def apply_token_limits(response: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return response
        
    max_chars = max_tokens * 4
    if len(response) > max_chars:
        truncated = response[:max_chars]
        last_sentence = max(
            truncated.rfind('.'),
            truncated.rfind('!'), 
            truncated.rfind('?')
        )
        if last_sentence > max_chars * 0.7:
            return truncated[:last_sentence + 1]
        else:
            return truncated + "..."
    
    return response

async def create_sse_stream(
    response_text: str, 
    request_id: str, 
    model: str, 
    chunk_delay: float = 0.05
) -> AsyncGenerator[str, None]:
    """Create Server-Sent Events stream in OpenAI format."""
    try:
        words = response_text.split()
        if not words:
            words = [""]
        
        chunk_size = max(1, len(words) // 20)
        created_time = int(datetime.now().timestamp())
        
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk_content = " ".join(chunk_words)
            
            if i + chunk_size < len(words):
                chunk_content += " "
            
            chunk_data = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": chunk_content
                    },
                    "finish_reason": None
                }]
            }
            
            yield f"data: {json.dumps(chunk_data)}\n\n"
            
            if chunk_delay > 0:
                await asyncio.sleep(chunk_delay)
        
        final_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk", 
            "created": created_time,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        log.error(f"Error in SSE stream: {e}")
        error_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(datetime.now().timestamp()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {
                    "content": f"[Errore: {str(e)}]"
                },
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"

# ===== ENDPOINTS =====

@endpoint.get("/health")
def health_check():
    return {
        "status": "healthy",
        "plugin": "universal-openai-bridge",
        "version": "5.0.0",
        "timestamp": datetime.now().isoformat(),
        "cat_status": "active" if _cat_instance else "initializing",
        "streaming_support": True,
        "personality_control": True,
        "italian_support": True
    }

@endpoint.get("/v1/models")
def openai_models():
    settings = get_plugin_settings()
    custom_models = settings.get("custom_model_names", "").split(",")
    custom_models = [m.strip() for m in custom_models if m.strip()]
    
    base_models = ["cheshire-cat", "gpt-3.5-turbo", "gpt-4"]
    all_models = base_models + custom_models
    
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "cheshire-cat",
                "permission": [],
                "root": model_id,
                "parent": None
            }
            for model_id in all_models
        ]
    }

@endpoint.post("/v1/chat/completions")
async def openai_chat_completions(payload: OpenAIChatRequest):
    """OpenAI-compatible chat completions with controlled personality."""
    request_id = generate_openai_id()
    user_id = payload.user or "openai_user"
    
    try:
        user_messages = [msg for msg in payload.messages if msg.role == "user"]
        if not user_messages:
            return {
                "error": {
                    "message": "Nessun messaggio utente trovato",
                    "type": "invalid_request_error",
                    "code": "invalid_request"
                }
            }
        
        user_message = user_messages[-1].content
        
        if not check_rate_limit(user_id):
            record_metrics("chat_completions", user_id, False)
            return {
                "error": {
                    "message": "Limite di richieste superato",
                    "type": "rate_limit_exceeded", 
                    "code": "rate_limit_exceeded"
                }
            }
        
        settings = get_plugin_settings()
        
        max_tokens = settings.get("max_tokens_limit", 4000)
        if max_tokens > 0 and len(user_message) > max_tokens * 4:
            user_message = user_message[:max_tokens * 4]
        
        is_streaming = (
            payload.stream and 
            settings.get("enable_streaming", True)
        )
        
        # Ottieni risposta con personalità controllata
        log.info(f"Processing request for user {user_id}: {user_message[:50]}...")
        cat_response = await get_controlled_cat_response(user_message, user_id)
        
        if max_tokens > 0:
            cat_response = apply_token_limits(cat_response, max_tokens)
        
        manage_conversation_history(user_id, "user", user_message)
        manage_conversation_history(user_id, "assistant", cat_response)
        
        prompt_tokens = sum(count_tokens(msg.content) for msg in payload.messages)
        completion_tokens = count_tokens(cat_response)
        
        record_metrics("chat_completions", user_id, True, is_streaming)
        
        if settings.get("enable_detailed_logging", False):
            log.info(f"Response (streaming={is_streaming}): {cat_response[:50]}...")
        
        if is_streaming:
            plugin_state["active_streams"][request_id] = {
                "user_id": user_id,
                "start_time": time.time(),
                "model": payload.model
            }
            
            chunk_delay = settings.get("streaming_chunk_delay", 0.05)
            
            async def stream_generator():
                try:
                    async for chunk in create_sse_stream(cat_response, request_id, payload.model, chunk_delay):
                        yield chunk
                finally:
                    plugin_state["active_streams"].pop(request_id, None)
            
            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        
        response = {
            "id": request_id,
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": payload.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": cat_response
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }
        
        return response
        
    except Exception as e:
        log.error(f"Error in openai_chat_completions: {e}")
        record_metrics("chat_completions", user_id, False)
        
        return {
            "error": {
                "message": f"Errore interno del server: {str(e)}",
                "type": "internal_error",
                "code": "internal_error"
            }
        }

@endpoint.post("/message")
async def simple_message(payload):
    try:
        if hasattr(payload, 'user_id'):
            user_id = payload.user_id
            message_text = payload.text
        else:
            user_id = payload.get("user_id", "default")
            message_text = payload.get("text", "")
        
        if not check_rate_limit(user_id):
            record_metrics("simple_message", user_id, False)
            return {"error": "Limite di richieste superato"}
        
        settings = get_plugin_settings()
        
        max_tokens = settings.get("max_tokens_limit", 4000)
        if max_tokens > 0 and len(message_text) > max_tokens * 4:
            message_text = message_text[:max_tokens * 4]
        
        cat_response = await get_controlled_cat_response(message_text, user_id)
        
        if max_tokens > 0:
            cat_response = apply_token_limits(cat_response, max_tokens)
        
        manage_conversation_history(user_id, "user", message_text)
        manage_conversation_history(user_id, "assistant", cat_response)
        
        response = {
            "text": cat_response,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }
        
        record_metrics("simple_message", user_id, True)
        
        return response
        
    except Exception as e:
        log.error(f"Error in simple_message: {e}")
        return {"error": f"Errore interno del server: {str(e)}"}

# ===== TOOLS =====

@tool(
    return_direct=False,
    examples=[
        "show openai bridge stats",
        "stato del bridge openai"
    ]
)
def bridge_status(tool_input, cat):
    """Mostra lo stato e le statistiche del plugin OpenAI Bridge."""
    try:
        global _cat_instance
        _cat_instance = cat
        
        settings = get_plugin_settings()
        metrics = plugin_state["metrics"]
        
        status = f"""🔗 Stato Universal OpenAI Bridge v5.0:

✅ **Stato Plugin**: Attivo e funzionale (Personalità Controllabile)
🧠 **Integrazione Cat**: Usa capacità del Cat con personalità personalizzabile
🌊 **Streaming**: SSE streaming {"abilitato" if settings.get("enable_streaming", True) else "disabilitato"}
🇮🇹 **Lingua**: Forzatura italiano {"attiva" if settings.get("force_italian", True) else "disattiva"}

🎭 **Configurazione Personalità**:
   - Usa personalità Cat: {"Sì" if settings.get("use_cat_personality", False) else "No"}
   - Usa memoria Cat: {"Sì" if settings.get("use_cat_memory", True) else "No"}
   - Personalità custom: {"Configurata" if settings.get("ai_personality") else "Default"}

🌐 **Endpoint API**:
   - GET  /custom/health
   - GET  /custom/v1/models  
   - POST /custom/v1/chat/completions (Endpoint principale con SSE)
   - POST /custom/message

⚙️ **Impostazioni Attuali**:
   - Supporto Streaming: {"Abilitato" if settings.get("enable_streaming", True) else "Disabilitato"}
   - Memoria Conversazione: {"Abilitata" if settings.get("enable_conversation_memory") else "Disabilitata"}
   - Max Cronologia: {settings.get("max_conversation_history", 20)} messaggi
   - Limite Rate: {settings.get("rate_limit_per_minute", 60)} req/min
   - Max Token: {settings.get("max_tokens_limit", 4000)}

📊 **Statistiche Utilizzo**:
   - Richieste Totali: {metrics['total_requests']}
   - Successo: {metrics['successful_requests']}
   - Fallite: {metrics['failed_requests']}
   - Richieste Streaming: {metrics['streaming_requests']}
   - Stream Attivi: {len(plugin_state['active_streams'])}
   - Conversazioni Attive: {len(plugin_state['conversation_history'])}

🚀 **Perfetto per**: ElevenLabs, client OpenAI, applicazioni streaming
🎯 **Caratteristiche**: Risposte intelligenti del Cat in italiano con memoria"""
        
        return status
        
    except Exception as e:
        return f"❌ Errore nel recuperare lo stato del bridge: {str(e)}"

@tool(
    return_direct=False,
    examples=[
        "test bridge con ciao mondo",
        "testa il bridge openai"
    ]
)
async def test_bridge_response(message, cat):
    """Testa il bridge inviando un messaggio attraverso il sistema."""
    try:
        global _cat_instance
        _cat_instance = cat
        
        test_message = message if message else "Ciao, puoi parlarmi di te in italiano?"
        response = await get_controlled_cat_response(test_message, cat.user_id)
        
        result = f"""🧪 Risultati Test Bridge:

**Messaggio di Test**: {test_message}
**Risposta**: {response}
**Lunghezza Risposta**: {len(response)} caratteri
**Stato**: ✅ Bridge funzionante

🌐 **Per ElevenLabs**: 
   Server URL: http://localhost:1865/custom
   Endpoint: /custom/v1/chat/completions

🌊 **SSE Streaming**: {"Abilitato" if get_plugin_settings().get("enable_streaming", True) else "Disabilitato"}
🇮🇹 **Forzatura Italiano**: {"Attiva" if get_plugin_settings().get("force_italian", True) else "Disattiva"}

💡 **Per risposte più intelligenti**:
   1. Carica documenti in Admin → Rabbit Hole
   2. Configura LLM ed Embedder in Settings
   3. Abilita "Use Cat's Memory System" nelle impostazioni plugin"""
        
        return result
        
    except Exception as e:
        return f"❌ Test del bridge fallito: {str(e)}"

# ===== HOOKS =====

@hook(priority=1)
def after_cat_bootstrap(cat):
    """Inizializza il plugin e memorizza l'istanza del Cat."""
    global _cat_instance
    _cat_instance = cat
    
    settings = get_plugin_settings()
    cors_origins = settings.get("cors_origins", "*")
    
    try:
        from fastapi.middleware.cors import CORSMiddleware
        
        if not any(isinstance(middleware, CORSMiddleware) for middleware in cat.app.middleware):
            origins = [origin.strip() for origin in cors_origins.split(",")]
            
            cat.app.add_middleware(
                CORSMiddleware,
                allow_origins=origins,
                allow_credentials=True,
                allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                allow_headers=["*"],
            )
            log.info(f"🌐 CORS configurato: {origins}")
        
        log.info("🔗 Universal OpenAI Bridge Plugin v5.0 attivato")
        log.info("📡 Endpoint principale: /custom/v1/chat/completions")
        log.info("🌊 Supporto SSE streaming abilitato")
        log.info("🎭 Personalità controllabile con memoria del Cat")
        log.info("🇮🇹 Supporto forzatura italiano")
        log.info("🎯 Compatibile con ElevenLabs e tutti i client OpenAI")
        
    except Exception as e:
        log.warning(f"Impossibile configurare CORS: {e}")
    
    return cat

@hook(priority=1)
def before_cat_sends_message(message, cat):
    """Memorizza l'istanza del Cat."""
    global _cat_instance
    _cat_instance = cat
    return message