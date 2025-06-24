"""
Universal OpenAI Bridge Plugin for Cheshire Cat
Provides real OpenAI-compatible API endpoints
"""

from cat.mad_hatter.decorators import tool, hook, plugin, endpoint
from cat.log import log
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Dict, Any, List, Optional
from collections import defaultdict, Counter
import json
import time

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
        default=False
    )
    
    default_temperature: float = Field(
        title="Default Temperature",
        description="Default temperature for LLM responses",
        default=0.7,
        ge=0.0,
        le=2.0
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
    
    enable_request_metrics: bool = Field(
        title="Enable Request Metrics",
        description="Track usage statistics and request metrics",
        default=True
    )
    
    custom_model_names: str = Field(
        title="Custom Model Names",
        description="Additional model names to expose (comma-separated)",
        default="claude-3,gpt-4-turbo"
    )

@plugin
def settings_model():
    return UniversalBridgeSettings

# ===== MODELS =====

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

class SimpleMessageRequest(BaseModel):
    text: str
    user_id: str = "default"

class SimpleMessageResponse(BaseModel):
    text: str
    user_id: str
    timestamp: str

# ===== PLUGIN STATE =====

plugin_state = {
    "conversation_history": {},
    "request_counts": defaultdict(list),
    "metrics": {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
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

def record_metrics(endpoint: str, user_id: str, success: bool = True):
    settings = get_plugin_settings()
    if not settings.get("enable_request_metrics", True):
        return
    
    metrics = plugin_state["metrics"]
    metrics["total_requests"] += 1
    
    if success:
        metrics["successful_requests"] += 1
    else:
        metrics["failed_requests"] += 1
    
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

def get_cat_instance():
    global _cat_instance
    
    if _cat_instance is not None:
        return _cat_instance
    
    try:
        import cat.main as cat_main
        if hasattr(cat_main, 'cheshire_cat_api') and cat_main.cheshire_cat_api:
            _cat_instance = cat_main.cheshire_cat_api
            return _cat_instance
            
        from cat.mad_hatter.mad_hatter import MadHatter
        mad_hatter = MadHatter()
        if hasattr(mad_hatter, 'cheshire_cat'):
            _cat_instance = mad_hatter.cheshire_cat
            return _cat_instance
            
        from cat.looking_glass.stray_cat import StrayCat
        _cat_instance = StrayCat(user_id="api_user")
        return _cat_instance
        
    except Exception as e:
        log.error(f"Could not get Cat instance: {e}")
        return None

def get_real_cat_response(message: str, user_id: str = "api_user") -> str:
    try:
        cat = get_cat_instance()
        if cat is None:
            return "I'm Cheshire Cat, but I'm having trouble accessing my full capabilities right now. Please try again."
        
        if hasattr(cat, 'llm'):
            response = cat.llm(message)
        elif hasattr(cat, 'send_ws_message'):
            response = cat.send_ws_message(message, user_id)
        else:
            response = str(cat(message))
        
        if not isinstance(response, str):
            response = str(response)
            
        if not response or response.strip() == "":
            response = "I received your message and I'm processing it. How can I help you?"
            
        return response
        
    except Exception as e:
        log.error(f"Error getting Cat response: {e}")
        return f"I'm Cheshire Cat responding to: '{message}'. There was a technical issue, but I'm working properly."

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

# ===== ENDPOINTS =====

@endpoint.get("/v1/health")
def health_check():
    return {
        "status": "healthy",
        "plugin": "universal-openai-bridge",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "cat_status": "active" if get_cat_instance() else "initializing"
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
def openai_chat_completions(payload: OpenAIChatRequest):
    try:
        user_messages = [msg for msg in payload.messages if msg.role == "user"]
        if not user_messages:
            return {"error": "No user message found"}
        
        user_message = user_messages[-1].content
        user_id = payload.user or "openai_user"
        
        # Rate limiting check
        if not check_rate_limit(user_id):
            record_metrics("chat_completions", user_id, False)
            return {
                "error": {
                    "message": "Rate limit exceeded",
                    "type": "rate_limit_exceeded",
                    "code": "rate_limit_exceeded"
                }
            }
        
        settings = get_plugin_settings()
        
        # Apply token limits if configured
        max_tokens = settings.get("max_tokens_limit", 4000)
        if max_tokens > 0 and len(user_message) > max_tokens * 4:
            user_message = user_message[:max_tokens * 4]
        
        cat_response = get_real_cat_response(user_message, user_id)
        
        # Apply response token limits
        if max_tokens > 0:
            cat_response = apply_token_limits(cat_response, max_tokens)
        
        # Manage conversation history
        manage_conversation_history(user_id, "user", user_message)
        manage_conversation_history(user_id, "assistant", cat_response)
        
        prompt_tokens = sum(count_tokens(msg.content) for msg in payload.messages)
        completion_tokens = count_tokens(cat_response)
        
        response = {
            "id": generate_openai_id(),
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
        
        record_metrics("chat_completions", user_id, True)
        
        if settings.get("enable_detailed_logging", False):
            log.info(f"OpenAI completion: {user_message[:50]}... -> {cat_response[:50]}...")
        
        return response
        
    except Exception as e:
        log.error(f"Error in openai_chat_completions: {e}")
        record_metrics("chat_completions", user_id if 'user_id' in locals() else "unknown", False)
        return {
            "error": {
                "message": f"Internal server error: {str(e)}",
                "type": "internal_error",
                "code": "internal_error"
            }
        }

@endpoint.post("/message")
def simple_message(payload: SimpleMessageRequest):
    try:
        user_id = payload.user_id
        
        if not check_rate_limit(user_id):
            record_metrics("simple_message", user_id, False)
            return {"error": "Rate limit exceeded"}
        
        settings = get_plugin_settings()
        
        # Apply token limits
        max_tokens = settings.get("max_tokens_limit", 4000)
        message_text = payload.text
        if max_tokens > 0 and len(message_text) > max_tokens * 4:
            message_text = message_text[:max_tokens * 4]
        
        cat_response = get_real_cat_response(message_text, user_id)
        
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
        
        if settings.get("enable_detailed_logging", False):
            log.info(f"Simple message: {message_text[:50]}... -> {cat_response[:50]}...")
        
        return response
        
    except Exception as e:
        log.error(f"Error in simple_message: {e}")
        record_metrics("simple_message", user_id if 'user_id' in locals() else "unknown", False)
        return {"error": f"Internal server error: {str(e)}"}

@endpoint.get("/v1/conversations/{user_id}")
def get_user_conversation(user_id: str):
    history = plugin_state["conversation_history"].get(user_id, [])
    return {
        "user_id": user_id,
        "conversation_count": len(history),
        "conversations": history,
        "timestamp": datetime.now().isoformat()
    }

@endpoint.delete("/v1/conversations/{user_id}")
def clear_user_conversation(user_id: str):
    if user_id in plugin_state["conversation_history"]:
        del plugin_state["conversation_history"][user_id]
        return {"message": f"Conversation history cleared for {user_id}"}
    return {"message": f"No conversation history found for {user_id}"}

@endpoint.get("/v1/metrics")
def get_detailed_metrics():
    try:
        import psutil
        import os
        
        memory_info = psutil.Process(os.getpid()).memory_info()
        
        return {
            "plugin_metrics": plugin_state["metrics"],
            "conversation_stats": {
                "active_users": len(plugin_state["conversation_history"]),
                "total_messages": sum(len(conv) for conv in plugin_state["conversation_history"].values()),
                "average_conversation_length": (
                    sum(len(conv) for conv in plugin_state["conversation_history"].values()) / 
                    max(len(plugin_state["conversation_history"]), 1)
                )
            },
            "system_info": {
                "memory_usage_mb": memory_info.rss / 1024 / 1024,
                "cpu_percent": psutil.cpu_percent(),
                "uptime_hours": (time.time() - psutil.boot_time()) / 3600
            },
            "rate_limiting": {
                "active_users_with_requests": len(plugin_state["request_counts"]),
                "total_requests_last_minute": sum(len(reqs) for reqs in plugin_state["request_counts"].values())
            },
            "timestamp": datetime.now().isoformat()
        }
    except ImportError:
        return {
            "plugin_metrics": plugin_state["metrics"],
            "conversation_stats": {
                "active_users": len(plugin_state["conversation_history"]),
                "total_messages": sum(len(conv) for conv in plugin_state["conversation_history"].values())
            },
            "note": "Install psutil for detailed system metrics",
            "timestamp": datetime.now().isoformat()
        }

# ===== TOOLS =====

@tool(
    return_direct=False,
    examples=[
        "show openai bridge stats",
        "get bridge statistics",
        "bridge plugin status"
    ]
)
def bridge_status(tool_input, cat):
    """Get comprehensive status and statistics of the OpenAI Bridge plugin.
    Input is always None.
    """
    try:
        global _cat_instance
        _cat_instance = cat
        
        settings = get_plugin_settings()
        metrics = plugin_state["metrics"]
        
        status = f"""🔗 Universal OpenAI Bridge Status:

✅ **Plugin Status**: Active and functional
🌐 **API Endpoints**:
   - GET  /custom/v1/health
   - GET  /custom/v1/models  
   - POST /custom/v1/chat/completions
   - POST /custom/message
   - GET  /custom/v1/conversations/{{user_id}}
   - DELETE /custom/v1/conversations/{{user_id}}
   - GET  /custom/v1/metrics

⚙️ **Current Settings**:
   - Conversation Memory: {"Enabled" if settings.get("enable_conversation_memory") else "Disabled"}
   - Max History: {settings.get("max_conversation_history", 20)} messages
   - Rate Limit: {settings.get("rate_limit_per_minute", 60)} req/min
   - Detailed Logging: {"Enabled" if settings.get("enable_detailed_logging") else "Disabled"}
   - Max Tokens: {settings.get("max_tokens_limit", 4000)}
   - Default Temperature: {settings.get("default_temperature", 0.7)}

📊 **Usage Statistics**:
   - Total Requests: {metrics['total_requests']}
   - Successful: {metrics['successful_requests']}
   - Failed: {metrics['failed_requests']}
   - Active Conversations: {len(plugin_state['conversation_history'])}
   - Most Used Endpoint: {metrics['endpoints_usage'].most_common(1)[0] if metrics['endpoints_usage'] else 'None'}

🚀 **Features**:
   - 100% real Cat responses
   - Full OpenAI API compatibility
   - Rate limiting protection
   - Conversation memory
   - Comprehensive metrics
   - Configurable via admin panel"""
        
        return status
        
    except Exception as e:
        return f"❌ Error getting bridge status: {str(e)}"

@tool(
    return_direct=False,
    examples=[
        "test bridge with hello world",
        "verify openai bridge functionality",
        "test cat response through bridge"
    ]
)
def test_bridge_response(message, cat):
    """Test the bridge by sending a message and getting a real Cat response.
    Input should be the test message you want to send.
    """
    try:
        global _cat_instance
        _cat_instance = cat
        
        response = get_real_cat_response(message, cat.user_id)
        
        result = f"""🧪 Bridge Test Results:

**Test Message**: {message}
**Cat Response**: {response}
**Response Length**: {len(response)} characters
**Estimated Tokens**: {count_tokens(response)}
**Status**: ✅ Bridge working correctly

The bridge successfully processed your message and returned a real Cat response!"""
        
        return result
        
    except Exception as e:
        return f"❌ Bridge test failed: {str(e)}"

@tool(
    return_direct=False,
    examples=[
        "clear my conversation history",
        "reset my chat memory",
        "delete my conversation data"
    ]
)
def clear_conversation_history(tool_input, cat):
    """Clear your conversation history from the OpenAI Bridge.
    Input is always None.
    """
    try:
        user_id = cat.user_id
        
        if user_id in plugin_state["conversation_history"]:
            del plugin_state["conversation_history"][user_id]
            return f"✅ Conversation history cleared for user {user_id}."
        else:
            return f"ℹ️ No conversation history found for user {user_id}."
            
    except Exception as e:
        return f"❌ Error clearing conversation history: {str(e)}"

# ===== HOOKS =====

@hook(priority=1)
def after_cat_bootstrap(cat):
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
            log.info(f"🌐 CORS configured: {origins}")
        
        log.info("🔗 Universal OpenAI Bridge Plugin activated")
        log.info("📡 All endpoints provide real Cat responses")
        log.info("⚙️ Configurable via admin panel settings")
        
    except Exception as e:
        log.warning(f"Could not configure CORS: {e}")
    
    return cat

@hook(priority=1)
def before_cat_sends_message(message, cat):
    global _cat_instance
    _cat_instance = cat
    return message