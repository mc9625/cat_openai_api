# 🔗 Universal OpenAI Bridge Plugin

**Transform your Cheshire Cat into a fully OpenAI-compatible API endpoint**

[![Cheshire Cat AI](https://img.shields.io/badge/Cheshire%20Cat%20AI-Plugin-purple)](https://github.com/cheshire-cat-ai/core)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-1.0.0-blue)](https://github.com/yourusername/universal-openai-bridge)

A comprehensive plugin that provides **real OpenAI-compatible API endpoints** for your Cheshire Cat, enabling seamless integration with any OpenAI-compatible application while delivering authentic Cat responses.

## 🌟 Features

### 🔌 **100% OpenAI API Compatibility**
- Full `/v1/chat/completions` endpoint compatibility
- `/v1/models` endpoint with custom model definitions
- Standard OpenAI request/response formats
- Support for all major OpenAI client libraries

### 🧠 **Real Cat Intelligence**
- **Authentic responses** directly from your Cheshire Cat
- Access to Cat's full knowledge base and memory
- Plugin ecosystem integration
- No external API calls - everything runs locally

### 🛡️ **Enterprise-Ready Features**
- **Rate limiting** with configurable limits per user
- **Conversation memory** with persistent context
- **Comprehensive metrics** and usage analytics
- **CORS support** for web applications
- **Token limiting** and content filtering

### ⚙️ **Highly Configurable**
- Web-based admin panel configuration
- Hot-reload settings without restart
- Custom model names and descriptions
- Flexible logging and debugging options

## 🚀 Quick Start

### Installation

1. **Download the Plugin**
   ```bash
   # Option 1: From Cat's Admin Panel
   # Go to Plugins > Plugin Registry > Search "Universal OpenAI Bridge"
   
   # Option 2: Manual Installation
   git clone https://github.com/mc9625/cat_openai_api.git
   # Copy to your Cat's plugins folder
   ```

2. **Activate the Plugin**
   - Navigate to `http://localhost:1865/admin`
   - Go to **Plugins** → **Installed Plugins**
   - Enable **Universal OpenAI Bridge**

3. **Configure Settings**
   - Click the ⚙️ icon next to the plugin
   - Adjust settings according to your needs
   - Save configuration

### Basic Usage

Once installed, your Cat will expose OpenAI-compatible endpoints:

```bash
# Health check
curl http://localhost:1865/custom/v1/health

# List available models
curl http://localhost:1865/custom/v1/models

# Chat completion (OpenAI format)
curl -X POST http://localhost:1865/custom/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cheshire-cat",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## 📋 API Documentation

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/custom/v1/health` | GET | Plugin health status |
| `/custom/v1/models` | GET | Available models list |
| `/custom/v1/chat/completions` | POST | OpenAI-compatible chat |
| `/custom/message` | POST | Simplified message endpoint |
| `/custom/v1/conversations/{user_id}` | GET | User conversation history |
| `/custom/v1/conversations/{user_id}` | DELETE | Clear user history |
| `/custom/v1/metrics` | GET | Detailed usage metrics |

### Chat Completions API

**POST** `/custom/v1/chat/completions`

```json
{
  "model": "cheshire-cat",
  "messages": [
    {"role": "user", "content": "What is the meaning of life?"}
  ],
  "max_tokens": 1000,
  "temperature": 0.7,
  "user": "user123"
}
```

**Response:**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "cheshire-cat",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "The meaning of life, according to my feline wisdom..."
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 50,
    "total_tokens": 62
  }
}
```

### Simple Message API

**POST** `/custom/message`

```json
{
  "text": "Hello, Cat!",
  "user_id": "user123"
}
```

**Response:**
```json
{
  "text": "Hello! I'm your Cheshire Cat, ready to help!",
  "user_id": "user123",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## ⚙️ Configuration

### Plugin Settings

Access settings via Admin Panel → Plugins → Universal OpenAI Bridge ⚙️

| Setting | Default | Description |
|---------|---------|-------------|
| **Enable Conversation Memory** | `true` | Keep conversation context between requests |
| **Max Conversation History** | `20` | Maximum messages to remember per user |
| **Rate Limit (per minute)** | `60` | Maximum requests per minute per user |
| **Enable Detailed Logging** | `false` | Log detailed request/response information |
| **Enable Streaming** | `false` | Support streaming responses (experimental) |
| **Default Temperature** | `0.7` | Default LLM temperature for responses |
| **Max Tokens Limit** | `4000` | Maximum tokens per request (0 = no limit) |
| **CORS Origins** | `*` | Allowed CORS origins (comma-separated) |
| **Enable Request Metrics** | `true` | Track usage statistics |
| **Custom Model Names** | `claude-3,gpt-4-turbo` | Additional model names to expose |

### Advanced Configuration

For production deployments, consider these settings:

```json
{
  "enable_conversation_memory": true,
  "max_conversation_history": 50,
  "rate_limit_per_minute": 120,
  "enable_detailed_logging": false,
  "max_tokens_limit": 8000,
  "cors_origins": "https://yourapp.com,https://api.yourapp.com",
  "enable_request_metrics": true
}
```

## 🔧 Client Examples

### Python (OpenAI Library)

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:1865/custom/v1",
    api_key="not-needed"  # Cat doesn't require API keys
)

response = client.chat.completions.create(
    model="cheshire-cat",
    messages=[
        {"role": "user", "content": "Explain quantum physics"}
    ]
)

print(response.choices[0].message.content)
```

### JavaScript/Node.js

```javascript
import OpenAI from 'openai';

const openai = new OpenAI({
  baseURL: 'http://localhost:1865/custom/v1',
  apiKey: 'not-needed'
});

const completion = await openai.chat.completions.create({
  messages: [{ role: 'user', content: 'Hello Cat!' }],
  model: 'cheshire-cat',
});

console.log(completion.choices[0].message.content);
```

### Curl

```bash
# Simple chat
curl -X POST http://localhost:1865/custom/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cheshire-cat",
    "messages": [
      {"role": "user", "content": "Tell me a joke"}
    ],
    "temperature": 0.9
  }'
```

### LangChain Integration

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:1865/custom/v1",
    api_key="not-needed",
    model="cheshire-cat"
)

response = llm.invoke("What can you do for me?")
print(response.content)
```

## 🛠️ Built-in Tools

The plugin includes useful tools accessible via chat:

### 🔍 **Bridge Status**
```
"show openai bridge stats"
"get bridge statistics"  
"bridge plugin status"
```
Get comprehensive status and usage statistics.

### 🧪 **Test Bridge**
```
"test bridge with hello world"
"verify openai bridge functionality"
```
Test the bridge functionality with a sample message.

### 🗑️ **Clear History**
```
"clear my conversation history"
"reset my chat memory"
```
Clear your conversation history from the bridge.

## 📊 Monitoring & Metrics

### Health Monitoring

```bash
# Check plugin health
curl http://localhost:1865/custom/v1/health

# Get detailed metrics
curl http://localhost:1865/custom/v1/metrics
```

### Metrics Dashboard

The `/v1/metrics` endpoint provides:

- **Request Statistics**: Total, successful, failed requests
- **User Activity**: Active users, conversation lengths
- **Rate Limiting**: Current usage per user
- **System Info**: Memory usage, CPU, uptime
- **Endpoint Usage**: Most popular endpoints

Example response:
```json
{
  "plugin_metrics": {
    "total_requests": 1250,
    "successful_requests": 1240,
    "failed_requests": 10,
    "endpoints_usage": {
      "chat_completions": 800,
      "simple_message": 450
    }
  },
  "conversation_stats": {
    "active_users": 25,
    "total_messages": 5000,
    "average_conversation_length": 12.5
  }
}
```

## 🔐 Security & Best Practices

### Rate Limiting
- Configure appropriate rate limits for your use case
- Monitor the `/v1/metrics` endpoint for abuse patterns
- Consider implementing IP-based limiting for public deployments

### CORS Configuration
```json
{
  "cors_origins": "https://yourapp.com,https://api.yourapp.com"
}
```

### Token Limits
- Set `max_tokens_limit` to prevent resource exhaustion
- Monitor conversation memory usage
- Clear old conversations periodically

### Production Deployment
- Disable detailed logging in production
- Use reverse proxy (nginx) for SSL termination
- Monitor system resources and set appropriate limits
- Implement backup strategies for conversation data

## 🛠️ Troubleshooting

### Common Issues

**Q: Bridge returns "I'm having trouble accessing my full capabilities"**
- A: The plugin couldn't access the Cat instance. Restart the Cat and wait for full initialization.

**Q: Rate limiting is too strict/loose**
- A: Adjust `rate_limit_per_minute` in plugin settings. Set to 0 to disable.

**Q: CORS errors in web applications**
- A: Configure `cors_origins` with your domain or use `*` for development.

**Q: Responses are truncated**
- A: Increase `max_tokens_limit` or set to 0 for no limit.

**Q: High memory usage**
- A: Reduce `max_conversation_history` or disable conversation memory.

### Debug Mode

Enable detailed logging in plugin settings to troubleshoot issues:

```json
{
  "enable_detailed_logging": true
}
```

Check Cat logs for detailed request/response information.

### Performance Optimization

1. **Conversation Memory**: Reduce history length for high-traffic scenarios
2. **Rate Limiting**: Set appropriate limits based on your hardware
3. **Token Limits**: Prevent extremely long requests/responses
4. **Metrics**: Disable if not needed to reduce overhead

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the Repository**
2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make Your Changes**
4. **Test Thoroughly**
   - Test all endpoints
   - Verify OpenAI compatibility
   - Check error handling
5. **Submit a Pull Request**

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/universal-openai-bridge.git

# Install in development mode
ln -s $(pwd)/universal-openai-bridge /path/to/cat/plugins/

# Enable detailed logging for development
# Set enable_detailed_logging: true in plugin settings
```

### Code Style
- Follow PEP 8 for Python code
- Use type hints where appropriate
- Include docstrings for all functions
- Add examples to tool decorators

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Cheshire Cat AI](https://github.com/cheshire-cat-ai/core) - The amazing framework that makes this possible
- [OpenAI](https://openai.com/) - For the API standard that enables universal compatibility
- The Cheshire Cat community for feedback and contributions

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/mc9625/cat_openai_api/issues)
- **Discussions**: [GitHub Discussions](https://github.com/mc9625/cat_openai_api/discussions)
- **Discord**: [Cheshire Cat AI Discord](https://discord.gg/bHX5sNFCYU)

---

**Made with 🧩 for the Cheshire Cat AI ecosystem**

> "We're all mad here. But now we're OpenAI-compatible mad!" 🎩✨