"""
OpenAI API Bridge Plugin for Cheshire Cat AI

Provides OpenAI-compatible API endpoints that use the complete Cat pipeline
including RAG, memory, hooks, and personality. Perfect for integrating
third-party applications that expect OpenAI API format.
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

# ===== PLUGIN SETTINGS =====

class OpenAIBridgeSettings(BaseModel):
    """Configuration settings for the OpenAI Bridge plugin."""
    
    rate_limit_per_minute: int = 60
    max_tokens: Optional[int] = 4000
    enable_streaming: bool = True
    debug_mode: bool = False

@plugin
def settings_model():
    """Return the settings model for the plugin."""
    return OpenAIBridgeSettings

# ===== OPENAI API MODELS =====

class OpenAIMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str = "cheshire-cat"
    messages: List[OpenAIMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: Optional[bool] = False
    user: Optional[str] = None

# ===== PLUGIN STATE =====

# Storage for response interception
_pending_responses = {}
_request_counts = defaultdict(list)
_active_requests = set()

# ===== UTILITY FUNCTIONS =====

def generate_openai_id() -> str:
    """Generate a unique OpenAI-style request ID."""
    timestamp = int(time.time())
    import random, string
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=12))
    return f"chatcmpl-{timestamp}{random_suffix}"

def check_rate_limit(user_id: str) -> bool:
    """Check if user has exceeded rate limit."""
    settings = get_plugin_settings()
    rate_limit = settings.get("rate_limit_per_minute", 60)
    
    if rate_limit <= 0:
        return True
    
    now = time.time()
    minute_ago = now - 60
    
    _request_counts[user_id] = [
        req_time for req_time in _request_counts[user_id] 
        if req_time > minute_ago
    ]
    
    if len(_request_counts[user_id]) >= rate_limit:
        return False
    
    _request_counts[user_id].append(now)
    return True

def get_plugin_settings() -> Dict:
    """Get current plugin settings."""
    try:
        # This would be implemented based on Cat's plugin system
        return {
            "rate_limit_per_minute": 60,
            "max_tokens": 4000,
            "enable_streaming": True,
            "debug_mode": False
        }
    except:
        return {}

def count_tokens(text: str) -> int:
    """Estimate token count (simple approximation)."""
    return max(1, len(text) // 4)

def create_sse_chunk(request_id: str, content: str, model: str, finish_reason: Optional[str] = None) -> str:
    """Create a Server-Sent Events chunk in OpenAI format."""
    
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
    """Create the final SSE chunk."""
    return "data: [DONE]\n\n"

# ===== RESPONSE INTERCEPTION HOOK =====

@hook(priority=-9999)
def before_cat_sends_message(message: CatMessage, cat):
    """
    Intercept Cat responses for OpenAI API requests.
    
    This hook captures the final Cat response after the complete pipeline
    (including RAG, memory, hooks, and personality) has processed the message.
    """
    
    actual_user_id = cat.user_id
    message_text = message.text
    
    settings = get_plugin_settings()
    if settings.get("debug_mode", False):
        log.info(f"[OpenAI Bridge] Intercepting message from user '{actual_user_id}': {message_text[:100]}...")
    
    # Find any waiting request (dynamic user_id matching)
    found_request = None
    for req_id, data in _pending_responses.items():
        if data.get("waiting"):
            if settings.get("debug_mode", False):
                log.info(f"[OpenAI Bridge] Found waiting request {req_id}, updating user_id to '{actual_user_id}'")
            found_request = req_id
            data["actual_user_id"] = actual_user_id
            break
    
    if found_request:
        # Save the complete Cat response
        _pending_responses[found_request]["response"] = message_text
        _pending_responses[found_request]["intercepted"] = True
        _pending_responses[found_request]["waiting"] = False
        
        if settings.get("debug_mode", False):
            log.info(f"[OpenAI Bridge] Response intercepted for request {found_request}")
    
    return message

# ===== CAT PIPELINE INTEGRATION =====

async def get_cat_response_via_pipeline(prompt: str, request_user_id: str, request_id: str, cat_instance) -> str:
    """
    Get response from Cat using the complete pipeline.
    
    This ensures all Cat features work: RAG, memory, hooks, personality, etc.
    No fallbacks - if this fails, the request fails.
    """
    
    settings = get_plugin_settings()
    
    if request_id in _active_requests:
        raise Exception(f"Request {request_id} already being processed")
    
    _active_requests.add(request_id)
    
    try:
        # Register for response interception
        _pending_responses[request_id] = {
            "waiting": True,
            "request_user_id": request_user_id,
            "timestamp": time.time()
        }
        
        if settings.get("debug_mode", False):
            log.info(f"[OpenAI Bridge] Starting Cat pipeline for request {request_id}")
        
        # Create UserMessage exactly like the web interface does
        user_message = UserMessage(
            user_id=request_user_id,
            when=time.time(),
            who="Human", 
            text=prompt
        )
        
        # Send through Cat's complete pipeline
        if hasattr(cat_instance, 'receive'):
            cat_instance.receive(user_message)
        elif hasattr(cat_instance, '__call__'):
            cat_instance({
                "text": prompt,
                "user_id": request_user_id
            })
        else:
            raise Exception("Cannot access Cat pipeline")
        
        # Wait for response interception
        max_wait = 15
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            if request_id in _pending_responses:
                response_data = _pending_responses[request_id]
                
                if response_data.get("intercepted"):
                    cat_response = response_data["response"]
                    
                    # Cleanup
                    del _pending_responses[request_id]
                    _active_requests.discard(request_id)
                    
                    if settings.get("debug_mode", False):
                        log.info(f"[OpenAI Bridge] Pipeline response received for {request_id}")
                    
                    return cat_response
            
            await asyncio.sleep(0.1)
        
        # Timeout - this is an error, no fallback
        _pending_responses.pop(request_id, None)
        _active_requests.discard(request_id)
        
        raise Exception("Cat pipeline timeout - no response received")
        
    except Exception as e:
        # Cleanup on any error
        _pending_responses.pop(request_id, None)
        _active_requests.discard(request_id)
        raise e

# ===== STREAMING RESPONSE GENERATOR =====

async def stream_cat_response(prompt: str, user_id: str, request_id: str, model: str, cat_instance) -> AsyncGenerator[str, None]:
    """
    Generate streaming response using Cat's complete pipeline.
    """
    try:
        settings = get_plugin_settings()
        
        if settings.get("debug_mode", False):
            log.info(f"[OpenAI Bridge] Starting stream for request {request_id}")
        
        # Get complete response via Cat pipeline
        full_response = await get_cat_response_via_pipeline(prompt, user_id, request_id, cat_instance)
        
        # Stream the response word by word
        words = full_response.split()
        
        # Initial empty chunk
        yield create_sse_chunk(request_id, "", model)
        
        for word in words:
            yield create_sse_chunk(request_id, word + " ", model)
            await asyncio.sleep(0.03)  # Natural typing speed
        
        # Final chunk
        yield create_sse_chunk(request_id, "", model, "stop")
        yield create_sse_done()
        
        if settings.get("debug_mode", False):
            log.info(f"[OpenAI Bridge] Stream completed for {request_id}")
        
    except Exception as e:
        log.error(f"[OpenAI Bridge] Stream error for {request_id}: {e}")
        yield create_sse_chunk(request_id, f"Error: {str(e)}", model, "stop")
        yield create_sse_done()

# ===== API ENDPOINTS =====

@endpoint.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "plugin": "openai-bridge",
        "version": "1.0.0",
        "timestamp": int(time.time()),
        "features": ["chat_completions", "streaming", "rate_limiting"]
    }

@endpoint.get("/v1/models")
def list_models(
    stray = check_permissions(AuthResource.LLM, AuthPermission.READ)
):
    """List available models (OpenAI API compatible)."""
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
            }
        ]
    }

@endpoint.post("/v1/chat/completions")
async def chat_completions(
    body: Dict,
    stray = check_permissions(AuthResource.CONVERSATION, AuthPermission.WRITE)
):
    """
    OpenAI-compatible chat completions endpoint.
    
    Supports both streaming and non-streaming responses.
    All responses use the complete Cat pipeline.
    """
    request_id = generate_openai_id()
    user_id = body.get("user", "api_user")
    is_streaming = body.get("stream", False)
    
    try:
        settings = get_plugin_settings()
        
        if settings.get("debug_mode", False):
            log.info(f"[OpenAI Bridge] Request {request_id} from {user_id} (stream: {is_streaming})")
        
        # Validation
        if "messages" not in body or not body["messages"]:
            raise HTTPException(status_code=422, detail="messages field is required")

        # Rate limiting
        if not check_rate_limit(user_id):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        # Extract user message (latest only)
        user_messages = [msg for msg in body["messages"] if msg.get("role") == "user"]
        if not user_messages:
            raise HTTPException(status_code=422, detail="No user message found")

        user_message = user_messages[-1]["content"]
        model = body.get("model", "cheshire-cat")
        
        # Apply token limits
        max_tokens = body.get("max_tokens") or settings.get("max_tokens", 4000)
        if max_tokens > 0 and len(user_message) > max_tokens * 4:
            user_message = user_message[:max_tokens * 4]

        # Streaming response
        if is_streaming and settings.get("enable_streaming", True):
            return StreamingResponse(
                stream_cat_response(user_message, user_id, request_id, model, stray),
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive", 
                    "Content-Type": "text/plain; charset=utf-8"
                }
            )
        
        # Non-streaming response
        else:
            cat_response = await get_cat_response_via_pipeline(user_message, user_id, request_id, stray)

            # Apply response token limits
            if max_tokens > 0 and len(cat_response) > max_tokens * 4:
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

            return response

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"[OpenAI Bridge] Request {request_id} failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@endpoint.post("/message")
async def simple_message(
    body: Dict,
    stray = check_permissions(AuthResource.CONVERSATION, AuthPermission.WRITE)
):
    """Simple message endpoint for basic integrations."""
    try:
        text = body.get("text", "")
        user_id = body.get("user_id", "api_user")
        request_id = generate_openai_id()
        
        if not text.strip():
            raise HTTPException(status_code=422, detail="text field is required")

        # Rate limiting
        if not check_rate_limit(user_id):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        cat_response = await get_cat_response_via_pipeline(text, user_id, request_id, stray)

        return {
            "text": cat_response,
            "user_id": user_id,
            "timestamp": int(time.time())
        }

    except Exception as e:
        log.error(f"[OpenAI Bridge] Simple message error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# ===== PLUGIN INITIALIZATION =====

@hook(priority=1)
def after_cat_bootstrap(cat):
    """Initialize the OpenAI Bridge plugin."""
    try:
        log.info("🚀 OpenAI Bridge Plugin v1.0.0 activated!")
        log.info("📡 Available endpoints:")
        log.info("   - GET  /custom/health")
        log.info("   - GET  /custom/v1/models")
        log.info("   - POST /custom/v1/chat/completions")
        log.info("   - POST /custom/message")
        log.info("✨ Features: Complete Cat pipeline, RAG, memory, streaming")
        
    except Exception as e:
        log.error(f"Error initializing OpenAI Bridge: {e}")
    
    return cat