import asyncio
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Set
from pydantic import BaseModel, Field
from loguru import logger

from .agent_types import AgentType, MessageType


class AgentMessage(BaseModel):
    """Message structure for inter-agent communication"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    sender: AgentType
    recipient: Optional[AgentType] = None  # None for broadcast
    message_type: MessageType
    payload: Dict[str, Any] = Field(default_factory=dict)
    correlation_id: Optional[str] = None  # For request-response correlation
    priority: int = Field(default=5, ge=1, le=10)  # 1=highest, 10=lowest


class AgentResponse(BaseModel):
    """Response structure from agents"""
    success: bool
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MessageBus:
    """Central message bus for agent communication"""
    
    def __init__(self):
        self._subscribers: Dict[AgentType, List[Callable]] = {}
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._message_history: List[AgentMessage] = []
        self._max_history = 1000
        
    async def start(self):
        """Start the message bus"""
        self._running = True
        logger.info("Message bus started")
        
    async def stop(self):
        """Stop the message bus"""
        self._running = False
        logger.info("Message bus stopped")
        
    def subscribe(self, agent_type: AgentType, callback: Callable):
        """Subscribe an agent to receive messages"""
        if agent_type not in self._subscribers:
            self._subscribers[agent_type] = []
        self._subscribers[agent_type].append(callback)
        logger.debug(f"Agent {agent_type} subscribed to message bus")
        
    def unsubscribe(self, agent_type: AgentType, callback: Callable):
        """Unsubscribe an agent from receiving messages"""
        if agent_type in self._subscribers:
            try:
                self._subscribers[agent_type].remove(callback)
                logger.debug(f"Agent {agent_type} unsubscribed from message bus")
            except ValueError:
                pass
                
    async def publish(self, message: AgentMessage):
        """Publish a message to the bus"""
        self._store_message(message)
        
        # Handle broadcast messages
        if message.recipient is None:
            await self._broadcast(message)
        else:
            await self._send_to_recipient(message)
            
    async def _broadcast(self, message: AgentMessage):
        """Broadcast message to all subscribers except sender"""
        for agent_type, callbacks in self._subscribers.items():
            if agent_type != message.sender:
                for callback in callbacks:
                    try:
                        await callback(message)
                    except Exception as e:
                        logger.error(f"Error delivering message to {agent_type}: {e}")
                        
    async def _send_to_recipient(self, message: AgentMessage):
        """Send message to specific recipient"""
        if message.recipient in self._subscribers:
            for callback in self._subscribers[message.recipient]:
                try:
                    await callback(message)
                except Exception as e:
                    logger.error(f"Error delivering message to {message.recipient}: {e}")
        else:
            logger.warning(f"No subscribers for recipient {message.recipient}")
            
    def _store_message(self, message: AgentMessage):
        """Store message in history"""
        self._message_history.append(message)
        if len(self._message_history) > self._max_history:
            self._message_history.pop(0)
            
    def get_message_history(self, 
                          agent_type: Optional[AgentType] = None,
                          message_type: Optional[MessageType] = None,
                          limit: int = 100) -> List[AgentMessage]:
        """Get message history with optional filtering"""
        messages = self._message_history
        
        if agent_type:
            messages = [m for m in messages if m.sender == agent_type or m.recipient == agent_type]
            
        if message_type:
            messages = [m for m in messages if m.message_type == message_type]
            
        return messages[-limit:]