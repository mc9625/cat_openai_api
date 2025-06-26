"""
OpenAI Bridge Plugin - PIPELINE COMPLETO del Cat + Fix user_id
Usa il pipeline completo per RAG e personalità ma con user_id corretto
"""

from uuid import uuid4
import time
import asyncio
import json
from typing import Dict, List, Optional, AsyncGenerator
from collections import defaultdict
from pydantic import BaseModel

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from cat.mad_hatter.decorators import endpoint, plugin, hook
from cat.auth.permissions import check_permissions, AuthResource, AuthPermission
from cat.log import log
from cat.convo.messages import CatMessage, UserMessage

# ===== SETTINGS =====

class OpenAIBridgeSettings(BaseModel):
    """Settings per il plugin OpenAI Bridge."""
    
    rate_limit_per_minute: int = 60
    max_tokens: Optional[int] = 4000
    debug_mode: bool = True

@plugin
def settings_model():
    return OpenAIBridgeSettings

# ===== MODELS =====

class OpenAIMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str = "cat"
    messages: List[OpenAIMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: Optional[bool] = False
    user: Optional[str] = None

# ===== GLOBAL STATE =====

# Storage per intercettazione con USER_ID DINAMICO
_pending_responses = {}
_request_counts = defaultdict(list)
_active_requests = set()

# ===== UTILITY FUNCTIONS =====

def generate_openai_id() -> str:
    timestamp = int(time.time())
    import random, string
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=12))
    return f"chatcmpl-{timestamp}{random_suffix}"

def check_rate_limit(user_id: str) -> bool:
    """Verifica rate limiting."""
    now = time.time()
    minute_ago = now - 60
    
    _request_counts[user_id] = [
        req_time for req_time in _request_counts[user_id] 
        if req_time > minute_ago
    ]
    
    if len(_request_counts[user_id]) >= 60:
        return False
    
    _request_counts[user_id].append(now)
    return True

def count_tokens(text: str) -> int:
    """Stima approssimativa dei token."""
    return max(1, len(text) // 4)

def create_sse_chunk(request_id: str, content: str, model: str, finish_reason: Optional[str] = None) -> str:
    """Crea un chunk SSE nel formato OpenAI."""
    
    if finish_reason:
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason
                }
            ]
        }
    else:
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk", 
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": content},
                    "finish_reason": None
                }
            ]
        }
    
    return f"data: {json.dumps(chunk)}\n\n"

def create_sse_done() -> str:
    """Crea il chunk finale SSE."""
    return "data: [DONE]\n\n"

# ===== HOOK PER INTERCETTAZIONE CON USER_ID DINAMICO =====

@hook(priority=-9999)
def before_cat_sends_message(message: CatMessage, cat):
    """
    HOOK INTERCETTAZIONE: Cerca la richiesta per QUALSIASI user_id attivo
    """
    
    actual_user_id = cat.user_id
    message_text = message.text
    
    log.info(f"[BRIDGE HOOK] ⭐ Message from user '{actual_user_id}': {message_text[:100]}...")
    log.info(f"[BRIDGE HOOK] Active requests: {list(_pending_responses.keys())}")
    
    # Cerca richieste in attesa per QUALSIASI user_id
    found_request = None
    for req_id, data in _pending_responses.items():
        if data.get("waiting"):
            log.info(f"[BRIDGE HOOK] Found waiting request {req_id}, updating user_id to '{actual_user_id}'")
            found_request = req_id
            
            # ⭐ AGGIORNA IL USER_ID CON QUELLO REALE ⭐
            data["actual_user_id"] = actual_user_id
            break
    
    if found_request:
        log.info(f"[BRIDGE HOOK] ✅ INTERCETTATA risposta per request {found_request}")
        
        # Salva la risposta
        _pending_responses[found_request]["response"] = message_text
        _pending_responses[found_request]["intercepted"] = True
        _pending_responses[found_request]["waiting"] = False
        
        log.info(f"[BRIDGE HOOK] Risposta RAG salvata: {message_text[:100]}...")
    else:
        log.info(f"[BRIDGE HOOK] ❌ Nessuna richiesta in attesa trovata")
    
    return message

# ===== FUNZIONE PER PIPELINE COMPLETO =====

async def get_cat_response_with_rag(prompt: str, request_user_id: str, request_id: str, cat_instance) -> str:
    """
    PIPELINE COMPLETO del Cat con RAG, ma con gestione user_id corretta
    """
    try:
        log.info(f"[BRIDGE RAG] Starting full pipeline for request {request_id}")
        
        # Evita riprocessing
        if request_id in _active_requests:
            log.warning(f"[BRIDGE RAG] Request {request_id} already active")
            return "Request already being processed"
        
        _active_requests.add(request_id)
        
        # Registra per intercettazione (senza specificare user_id)
        _pending_responses[request_id] = {
            "waiting": True,
            "request_user_id": request_user_id,  # ID richiesta originale
            "timestamp": time.time()
        }
        
        log.info(f"[BRIDGE RAG] Registered request {request_id} for pipeline intercept")
        
        # ⭐ ACCESSO AL CAT CON PIPELINE COMPLETO ⭐
        
        # METODO 1: Prova con l'user_id della richiesta prima
        if hasattr(cat_instance, 'send_ws_message'):
            try:
                log.info(f"[BRIDGE RAG] Trying send_ws_message with user_id '{request_user_id}'")
                
                # Prova a impostare temporaneamente l'user_id
                original_user_id = getattr(cat_instance, 'user_id', None)
                cat_instance.user_id = request_user_id
                
                response = cat_instance.send_ws_message(prompt, request_user_id)
                
                # Ripristina user_id originale
                if original_user_id:
                    cat_instance.user_id = original_user_id
                
                if response and len(str(response).strip()) > 0:
                    log.info(f"[BRIDGE RAG] ✅ send_ws_message success: {str(response)[:100]}...")
                    
                    # Pulisci
                    _pending_responses.pop(request_id, None)
                    _active_requests.discard(request_id)
                    
                    return str(response)
                    
            except Exception as e:
                log.warning(f"[BRIDGE RAG] send_ws_message failed: {e}")
        
        # METODO 2: Usa UserMessage + receive() per pipeline completo
        try:
            log.info(f"[BRIDGE RAG] Using UserMessage + receive() method")
            
            # Crea UserMessage come fa l'interfaccia web
            user_message = UserMessage(
                user_id=request_user_id,
                when=time.time(),
                who="Human", 
                text=prompt
            )
            
            # Invia al Cat per processing completo
            if hasattr(cat_instance, 'receive'):
                cat_instance.receive(user_message)
                log.info(f"[BRIDGE RAG] Message sent via receive()")
            elif hasattr(cat_instance, '__call__'):
                cat_instance({
                    "text": prompt,
                    "user_id": request_user_id
                })
                log.info(f"[BRIDGE RAG] Message sent via __call__()")
            else:
                raise Exception("Cannot send message to Cat")
            
            # Aspetta intercettazione con timeout più breve
            max_wait = 10
            start_time = time.time()
            
            log.info(f"[BRIDGE RAG] Waiting for full pipeline response...")
            
            while time.time() - start_time < max_wait:
                if request_id in _pending_responses:
                    response_data = _pending_responses[request_id]
                    
                    if response_data.get("intercepted"):
                        cat_response = response_data["response"]
                        actual_user_id = response_data.get("actual_user_id", request_user_id)
                        
                        # Pulisci
                        del _pending_responses[request_id]
                        _active_requests.discard(request_id)
                        
                        log.info(f"[BRIDGE RAG] ✅ Got RAG response from user '{actual_user_id}': {cat_response[:100]}...")
                        return cat_response
                
                await asyncio.sleep(0.1)
            
            # Timeout - prova fallback
            log.warning(f"[BRIDGE RAG] ⏰ Pipeline timeout, trying LLM fallback...")
            
        except Exception as e:
            log.error(f"[BRIDGE RAG] Pipeline error: {e}")
        
        # FALLBACK: LLM diretto
        _pending_responses.pop(request_id, None)
        _active_requests.discard(request_id)
        
        if hasattr(cat_instance, 'llm'):
            log.info(f"[BRIDGE RAG] Fallback to direct LLM")
            return cat_instance.llm(prompt)
        
        return f"I'm the Cheshire Cat. Pipeline processing failed for: {prompt[:50]}..."
        
    except Exception as e:
        # Pulisci sempre
        _pending_responses.pop(request_id, None)
        _active_requests.discard(request_id)
        
        log.error(f"[BRIDGE RAG] Critical error: {e}")
        
        # Fallback finale
        if hasattr(cat_instance, 'llm'):
            return cat_instance.llm(prompt)
        
        return f"I'm the Cheshire Cat. Error: {prompt[:50]}..."

# ===== STREAMING GENERATOR =====

async def stream_cat_response(prompt: str, user_id: str, request_id: str, model: str, cat_instance) -> AsyncGenerator[str, None]:
    """
    Generator per streaming con PIPELINE COMPLETO
    """
    try:
        log.info(f"[BRIDGE STREAM] Starting RAG stream for {request_id}")
        
        # ⭐ USA PIPELINE COMPLETO CON RAG ⭐
        full_response = await get_cat_response_with_rag(prompt, user_id, request_id, cat_instance)
        
        log.info(f"[BRIDGE STREAM] Got RAG response: {full_response[:100]}...")
        
        # Stream word-by-word
        words = full_response.split()
        
        # Chunk iniziale
        yield create_sse_chunk(request_id, "", model)
        
        for i, word in enumerate(words):
            yield create_sse_chunk(request_id, word + " ", model)
            await asyncio.sleep(0.03)
        
        # Finalize
        yield create_sse_chunk(request_id, "", model, "stop")
        yield create_sse_done()
        
        log.info(f"[BRIDGE STREAM] ✅ RAG stream completed")
        
    except Exception as e:
        log.error(f"[BRIDGE STREAM] ❌ Stream error: {e}")
        yield create_sse_chunk(request_id, f"Error: {str(e)}", model, "stop")
        yield create_sse_done()

# ===== ENDPOINTS =====

@endpoint.get("/health")
def health_check():
    """Health check."""
    return {
        "status": "healthy",
        "plugin": "openai-bridge-rag-fixed",
        "version": "15.0.0",
        "strategy": "full_cat_pipeline_with_rag",
        "elevenlabs_compatible": True,
        "features": ["rag", "memory", "personality", "hooks"],
        "timestamp": int(time.time())
    }

@endpoint.get("/v1/models")
def list_models(
    stray = check_permissions(AuthResource.LLM, AuthPermission.READ)
):
    """Lista modelli."""
    return {
        "object": "list",
        "data": [
            {
                "id": "cheshire-cat",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "cheshire-cat",
                "permission": [],
                "root": "cheshire-cat",
                "parent": None
            },
            {
                "id": "gpt-3.5-turbo",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "cheshire-cat",
                "permission": [],
                "root": "gpt-3.5-turbo", 
                "parent": None
            },
            {
                "id": "gpt-4",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "cheshire-cat",
                "permission": [],
                "root": "gpt-4",
                "parent": None
            }
        ]
    }

@endpoint.post("/v1/chat/completions")
async def chat_completions(
    body: Dict,
    stray = check_permissions(AuthResource.CONVERSATION, AuthPermission.WRITE)
):
    """
    ENDPOINT PRINCIPALE: PIPELINE COMPLETO del Cat con RAG
    """
    request_id = generate_openai_id()
    user_id = body.get("user", "elevenlabs_user")
    is_streaming = body.get("stream", False)
    
    try:
        log.info(f"[BRIDGE MAIN] 🚀 Request {request_id} from {user_id} (stream: {is_streaming})")
        
        # Validazione
        if "messages" not in body or not body["messages"]:
            raise HTTPException(status_code=422, detail="messages field is required")

        # Rate limiting
        if not check_rate_limit(user_id):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        # Estrai SOLO ultimo messaggio user
        user_messages = [msg for msg in body["messages"] if msg.get("role") == "user"]
        if not user_messages:
            raise HTTPException(status_code=422, detail="No user message found")

        user_message = user_messages[-1]["content"]
        model = body.get("model", "cheshire-cat")
        
        log.info(f"[BRIDGE MAIN] Processing with RAG: {user_message[:100]}...")

        # ⭐ STREAMING con RAG ⭐
        if is_streaming:
            log.info(f"[BRIDGE MAIN] 📡 Starting RAG stream")
            
            return StreamingResponse(
                stream_cat_response(user_message, user_id, request_id, model, stray),
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive", 
                    "Content-Type": "text/plain; charset=utf-8",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "*"
                }
            )
        
        # ⭐ RISPOSTA NORMALE con RAG ⭐
        else:
            log.info(f"[BRIDGE MAIN] 💬 Getting RAG response")
            
            cat_response = await get_cat_response_with_rag(user_message, user_id, request_id, stray)

            # Applica limiti se necessario
            max_tokens = body.get("max_tokens")
            if max_tokens and len(cat_response) > max_tokens * 4:
                cat_response = cat_response[:max_tokens * 4] + "..."

            response = {
                "id": request_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
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
                    "prompt_tokens": count_tokens(user_message),
                    "completion_tokens": count_tokens(cat_response),
                    "total_tokens": count_tokens(user_message + cat_response)
                }
            }

            log.info(f"[BRIDGE MAIN] ✅ RAG Response: {cat_response[:50]}...")
            return response

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"[BRIDGE MAIN] ❌ Request {request_id} failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@endpoint.get("/debug")
def debug_status():
    """Debug dello stato."""
    return {
        "active_requests": list(_active_requests),
        "pending_responses": {
            req_id: {k: v for k, v in data.items() if k != "response"} 
            for req_id, data in _pending_responses.items()
        },
        "request_counts": dict(_request_counts),
        "timestamp": int(time.time()),
        "strategy": "full_pipeline_with_rag",
        "hook_enabled": True
    }

# ===== HOOK DI INIZIALIZZAZIONE =====

@hook(priority=1)
def after_cat_bootstrap(cat):
    """Inizializza il plugin."""
    try:
        log.info("🚀 OpenAI Bridge Plugin v15.0 - PIPELINE COMPLETO CON RAG!")
        log.info("✨ RIPRISTINATO:")
        log.info("   - ✅ RAG (Retrieval Augmented Generation)")
        log.info("   - ✅ System prompt personalizzato (Noesis)")
        log.info("   - ✅ Memoria episodica/dichiarativa")
        log.info("   - ✅ Hook personalizzati")
        log.info("   - ✅ Knowledge base specifica")
        log.info("🔧 FIXATO:")
        log.info("   - ✅ User_id mismatch problem")
        log.info("   - ✅ Hook intercettazione dinamica")
        log.info("📡 Endpoints:")
        log.info("   - GET  /custom/health")
        log.info("   - GET  /custom/v1/models")
        log.info("   - POST /custom/v1/chat/completions ⭐ (con RAG)")
        log.info("   - GET  /custom/debug")
        log.info("🎭 Noesis is BACK! Con personalità e knowledge completa!")
        
    except Exception as e:
        log.error(f"Error in after_cat_bootstrap: {e}")
    
    return cat