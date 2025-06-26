OpenAI Bridge Plugin for Cheshire Cat AI
A comprehensive plugin that provides OpenAI-compatible API endpoints for Cheshire Cat AI, enabling seamless integration with third-party applications that expect OpenAI API format.

🚀 Features
Complete Cat Pipeline Integration: All responses use the full Cheshire Cat pipeline including RAG, memory, hooks, and personality
OpenAI API Compatibility: Fully compatible with OpenAI's chat completions API
Streaming Support: Real-time streaming responses with Server-Sent Events (SSE)
Rate Limiting: Configurable per-user rate limiting
Authentication: Integrated with Cat's permission system
Error Handling: Robust error handling with proper HTTP status codes
No Fallbacks: Ensures all responses go through Cat's complete system
📋 Requirements
Cheshire Cat AI v1.0.0 or later
Python 3.8+
🔧 Installation
Download or clone this plugin into your Cat's plugins directory:
bash
cd /path/to/cheshire-cat/plugins
git clone https://github.com/mc9625/cat_openai_api.git
Restart your Cheshire Cat instance
The plugin will automatically activate and be available at the custom endpoints
⚙️ Configuration
Access the plugin settings through the Cat admin panel:

Settings Options
Setting	Description	Default	Range
rate_limit_per_minute	Maximum requests per user per minute	60	0-1000
max_tokens	Maximum tokens per request (0 = unlimited)	4000	0-32000
enable_streaming	Enable SSE streaming responses	true	true/false
debug_mode	Enable detailed logging	false	true/false
📡 Available Endpoints
All endpoints are available under the /custom prefix:

Health Check
GET /custom/health
List Models
GET /custom/v1/models
Chat Completions (Main Endpoint)
POST /custom/v1/chat/completions
Simple Message
POST /custom/message
🔧 Usage Examples
Basic Chat Completion
bash
curl -X POST http://localhost:1865/custom/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cheshire-cat",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "user": "user123"
  }'
Streaming Response
bash
curl -X POST http://localhost:1865/custom/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cheshire-cat",
    "messages": [
      {"role": "user", "content": "Tell me a story"}
    ],
    "stream": true,
    "user": "user123"
  }'
Simple Message
bash
curl -X POST http://localhost:1865/custom/message \
  -H "Content-Type: application/json" \
  -d '{
    "text": "What is the weather like?",
    "user_id": "user123"
  }'
🔗 Integration Examples
Python with OpenAI SDK
python
import openai

# Configure to use your Cat instance
openai.api_base = "http://localhost:1865/custom/v1"
openai.api_key = "dummy"  # Not used but required by SDK

response = openai.ChatCompletion.create(
    model="cheshire-cat",
    messages=[
        {"role": "user", "content": "Hello from Python!"}
    ],
    user="python_user"
)

print(response.choices[0].message.content)
JavaScript/Node.js
javascript
const response = await fetch('http://localhost:1865/custom/v1/chat/completions', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'cheshire-cat',
    messages: [
      { role: 'user', content: 'Hello from JavaScript!' }
    ],
    user: 'js_user'
  })
});

const data = await response.json();
console.log(data.choices[0].message.content);
Third-Party Applications
This plugin enables integration with any application that supports OpenAI API format:

Chatbots and Voice Assistants
Customer Service Platforms
Content Management Systems
Educational Platforms
Business Intelligence Tools
🛡️ Security and Permissions
The plugin respects Cat's built-in authentication and permission system:

Read permissions required for model listing
Write permissions required for chat completions
Rate limiting protects against abuse
Input validation prevents malicious requests
📊 Response Format
Non-Streaming Response
json
{
  "id": "chatcmpl-1234567890",
  "object": "chat.completion",
  "created": 1677649420,
  "model": "cheshire-cat",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! I'm the Cheshire Cat. How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 15,
    "total_tokens": 27
  }
}
Streaming Response
data: {"id":"chatcmpl-1234","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"}}]}

data: {"id":"chatcmpl-1234","object":"chat.completion.chunk","choices":[{"delta":{"content":" there!"}}]}

data: [DONE]
🔍 Troubleshooting
Common Issues
No response received
Check that your Cat instance is running
Verify the endpoint URL is correct
Check Cat logs for errors
Rate limit exceeded
Reduce request frequency
Increase rate limit in plugin settings
Use different user IDs to distribute load
Timeout errors
Check if Cat's pipeline is processing correctly
Verify your prompt doesn't trigger infinite loops
Check Cat's memory and hook configurations
Debug Mode
Enable debug mode in plugin settings to get detailed logs:

bash
# Check Cat logs for detailed information
docker logs cheshire_cat_core
🤝 Contributing
Fork the repository
Create a feature branch
Make your changes
Add tests if applicable
Submit a pull request
📄 License
MIT License - see LICENSE file for details

🐱 About Cheshire Cat AI
This plugin is designed for Cheshire Cat AI, an open-source conversational AI framework.

📞 Support
Issues: Report bugs and request features on GitHub
Documentation: Visit the Cheshire Cat AI documentation
Community: Join the Discord server for support and discussions

Made with ❤️ for the Cheshire Cat AI community


> "We're all mad here. But now we're OpenAI-compatible mad!" 🎩✨