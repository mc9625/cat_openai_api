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

# Cache settings to avoid repeated lookups
_settings_cache = {}
_cache_timestamp = 0
_cache_ttl = 60  # Cache for 60 seconds

def get_plugin_settings() -> Dict:
    """Get current plugin settings with caching."""
    global _settings_cache, _cache_timestamp
    
    current_time = time.time()
    if current_time - _cache_timestamp > _cache_ttl:
        try:
            # This would be implemented based on Cat's plugin system
            _settings_cache = {
                "rate_limit_per_minute": 60,
                "max_tokens": 4000,
                "enable_streaming": True,
                "debug_mode": False
            }
            _cache_timestamp = current_time
        except:
            _settings_cache = {}
    
    return _settings_cache

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

@hook(priority=-1000)  # Less extreme priority for better performance
def before_cat_sends_message(message: CatMessage, cat):
    """
    Intercept Cat responses for OpenAI API requests - SPEED OPTIMIZED.
    
    This hook captures the final Cat response after the complete pipeline
    (including RAG, memory, hooks, and personality) has processed the message.
    """
    
    # FAST PATH: Early exit if no pending requests
    if not _pending_responses:
        return message
    
    actual_user_id = cat.user_id
    message_text = message.text
    
    # OPTIMIZED: Find waiting request with early termination
    found_request = None
    for req_id, data in _pending_responses.items():
        if data.get("waiting"):
            found_request = req_id
            data["actual_user_id"] = actual_user_id
            break  # Early exit - take first match
    
    if found_request:
        # Save the complete Cat response - minimize operations
        response_data = _pending_responses[found_request]
        response_data["response"] = message_text
        response_data["intercepted"] = True
        response_data["waiting"] = False
    
    return message

# ===== CAT PIPELINE INTEGRATION =====

async def get_cat_response_via_pipeline(prompt: str, request_user_id: str, request_id: str, cat_instance) -> str:
    """
    Get response from Cat using the complete pipeline - SPEED OPTIMIZED.
    
    This ensures all Cat features work: RAG, memory, hooks, personality, etc.
    No fallbacks - if this fails, the request fails.
    """
    
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
        
        # OPTIMIZED: Shorter timeout and faster polling
        max_wait = 8  # Reduced from 15 to 8 seconds
        start_time = time.time()
        poll_interval = 0.005  # Reduced from 0.1 to 0.005 (5ms)
        
        while time.time() - start_time < max_wait:
            if request_id in _pending_responses:
                response_data = _pending_responses[request_id]
                
                if response_data.get("intercepted"):
                    cat_response = response_data["response"]
                    
                    # Cleanup
                    del _pending_responses[request_id]
                    _active_requests.discard(request_id)
                    
                    return cat_response
            
            await asyncio.sleep(poll_interval)
        
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
    Generate streaming response using Cat's complete pipeline - SPEED OPTIMIZED.
    """
    try:
        # Get complete response via Cat pipeline
        full_response = await get_cat_response_via_pipeline(prompt, user_id, request_id, cat_instance)
        
        # OPTIMIZED: Faster streaming with smaller chunks and reduced delays
        words = full_response.split()
        
        # Initial empty chunk
        yield create_sse_chunk(request_id, "", model)
        
        # Stream 2-3 words at a time for better flow
        chunk_size = 2
        for i in range(0, len(words), chunk_size):
            word_chunk = " ".join(words[i:i+chunk_size]) + " "
            yield create_sse_chunk(request_id, word_chunk, model)
            await asyncio.sleep(0.015)  # Reduced from 0.03 to 0.015 (faster streaming)
        
        # Final chunk
        yield create_sse_chunk(request_id, "", model, "stop")
        yield create_sse_done()
        
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
        "version": "1.1.0",  # Speed optimized version
        "timestamp": int(time.time()),
        "features": ["chat_completions", "streaming", "rate_limiting", "speed_optimized"],
        "optimizations": {
            "polling_interval_ms": 5,
            "timeout_seconds": 8,
            "settings_caching": True,
            "streaming_delay_ms": 15
        }
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
    OpenAI-compatible chat completions endpoint - SPEED OPTIMIZED.
    
    Supports both streaming and non-streaming responses.
    All responses use the complete Cat pipeline.
    """
    request_id = generate_openai_id()
    user_id = body.get("user", "api_user")
    is_streaming = body.get("stream", False)
    
    try:
        # FAST VALIDATION: Check required fields immediately
        messages = body.get("messages")
        if not messages:
            raise HTTPException(status_code=422, detail="messages field is required")

        # OPTIMIZED: Extract user message with single pass
        user_message = None
        for msg in reversed(messages):  # Start from end for latest message
            if msg.get("role") == "user":
                user_message = msg["content"]
                break
        
        if not user_message:
            raise HTTPException(status_code=422, detail="No user message found")

        # Rate limiting (cached settings)
        if not check_rate_limit(user_id):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        model = body.get("model", "cheshire-cat")
        
        # Apply token limits if specified
        max_tokens = body.get("max_tokens")
        if max_tokens and max_tokens > 0 and len(user_message) > max_tokens * 4:
            user_message = user_message[:max_tokens * 4]

        # STREAMING: Fast path for streaming requests
        if is_streaming:
            return StreamingResponse(
                stream_cat_response(user_message, user_id, request_id, model, stray),
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive", 
                    "Content-Type": "text/plain; charset=utf-8"
                }
            )
        
        # NON-STREAMING: Standard response
        cat_response = await get_cat_response_via_pipeline(user_message, user_id, request_id, stray)

        # Apply response limits
        if max_tokens and max_tokens > 0 and len(cat_response) > max_tokens * 4:
            cat_response = cat_response[:max_tokens * 4] + "..."

        # Build response object efficiently
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": cat_response
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": count_tokens(user_message),
                "completion_tokens": count_tokens(cat_response),
                "total_tokens": count_tokens(user_message + cat_response)
            }
        }

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
        log.info("🚀 OpenAI Bridge Plugin v1.0.0 - SPEED OPTIMIZED!")
        log.info("📡 Available endpoints:")
        log.info("   - GET  /custom/health")
        log.info("   - GET  /custom/v1/models")
        log.info("   - POST /custom/v1/chat/completions")
        log.info("   - POST /custom/message")
        log.info("⚡ Performance optimizations:")
        log.info("   - 5ms polling interval (was 100ms)")
        log.info("   - 8s timeout (was 15s)")
        log.info("   - Settings caching enabled")
        log.info("   - Fast streaming (15ms delays)")
        log.info("   - Optimized hook priority")
        log.info("✨ Features: Complete Cat pipeline, RAG, memory, streaming")
        
    except Exception as e:
        log.error(f"Error initializing OpenAI Bridge: {e}")
    
    return cat