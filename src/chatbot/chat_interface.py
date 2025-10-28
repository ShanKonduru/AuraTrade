"""
AuraTrade Trading Chatbot Interface
Web-based chat interface for trading assistant
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn
import json
import asyncio
from datetime import datetime
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chatbot.trading_chatbot import TradingChatbot

app = FastAPI(title="AuraTrade Chat Interface", version="1.0.0")

# Templates and static files
templates = Jinja2Templates(directory="src/chatbot/templates")
app.mount("/static", StaticFiles(directory="src/chatbot/static"), name="static")

# Global chatbot instance
trading_bot = TradingChatbot()

class ConnectionManager:
    """WebSocket connection manager"""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

@app.get("/", response_class=HTMLResponse)
async def chat_interface(request: Request):
    """Serve the chat interface"""
    return templates.TemplateResponse("chat.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time chat"""
    await manager.connect(websocket)
    
    # Send welcome message
    welcome_msg = {
        "type": "bot_message",
        "content": """🤖 **Welcome to AuraTrade Trading Assistant!**

I'm here to help you with:
📊 Stock analysis and recommendations
💰 Price checks and market data  
🛡️ Risk management and stop losses
📈 Growth and dividend stock ideas
📰 Market news and trends

**Try asking:**
• "Analyze Apple stock"
• "Should I buy Tesla?"
• "What are some good dividend stocks?"
• "Calculate stop loss for my Microsoft position"

How can I help you today?""",
        "timestamp": datetime.now().isoformat()
    }
    await manager.send_personal_message(json.dumps(welcome_msg), websocket)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data["type"] == "user_message":
                user_input = message_data["content"].strip()
                
                if user_input:
                    # Echo user message
                    user_msg = {
                        "type": "user_message",
                        "content": user_input,
                        "timestamp": datetime.now().isoformat()
                    }
                    await manager.send_personal_message(json.dumps(user_msg), websocket)
                    
                    # Send typing indicator
                    typing_msg = {
                        "type": "typing",
                        "timestamp": datetime.now().isoformat()
                    }
                    await manager.send_personal_message(json.dumps(typing_msg), websocket)
                    
                    # Process with chatbot
                    try:
                        bot_response = await trading_bot.process_message(user_input)
                        
                        # Send bot response
                        bot_msg = {
                            "type": "bot_message",
                            "content": bot_response,
                            "timestamp": datetime.now().isoformat()
                        }
                        await manager.send_personal_message(json.dumps(bot_msg), websocket)
                        
                    except Exception as e:
                        error_msg = {
                            "type": "bot_message",
                            "content": f"❌ Sorry, I encountered an error: {str(e)}\n\n💡 Please try again or rephrase your question.",
                            "timestamp": datetime.now().isoformat()
                        }
                        await manager.send_personal_message(json.dumps(error_msg), websocket)
                        
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "AuraTrade Chat Interface",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/chat/history")
async def get_chat_history():
    """Get chat conversation history"""
    return {
        "messages": [
            {
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "is_user": msg.is_user,
                "intent": msg.intent,
                "entities": msg.entities
            }
            for msg in trading_bot.conversation_history[-50:]  # Last 50 messages
        ]
    }

@app.post("/api/chat/clear")
async def clear_chat_history():
    """Clear chat conversation history"""
    trading_bot.conversation_history.clear()
    trading_bot.user_context.clear()
    return {"status": "cleared"}

if __name__ == "__main__":
    uvicorn.run(
        "chat_interface:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )