import asyncio
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from loguru import logger

from .message_bus import MessageBus, AgentMessage, AgentResponse
from .agent_types import AgentType, MessageType


class BaseAgent(ABC):
    """Base class for all agents in the AuraTrade system"""
    
    def __init__(self, 
                 agent_type: AgentType,
                 message_bus: MessageBus,
                 config: Optional[Dict[str, Any]] = None):
        self.agent_type = agent_type
        self.message_bus = message_bus
        self.config = config or {}
        self.is_running = False
        self.agent_id = str(uuid.uuid4())
        
        # Performance metrics
        self.messages_processed = 0
        self.errors_count = 0
        self.last_heartbeat = datetime.now()
        
        # Subscribe to message bus
        self.message_bus.subscribe(self.agent_type, self._handle_message)
        
        logger.info(f"Initialized {self.agent_type} agent with ID: {self.agent_id}")
        
    async def start(self):
        """Start the agent"""
        self.is_running = True
        await self._initialize()
        logger.info(f"{self.agent_type} agent started")
        
    async def stop(self):
        """Stop the agent"""
        self.is_running = False
        await self._cleanup()
        self.message_bus.unsubscribe(self.agent_type, self._handle_message)
        logger.info(f"{self.agent_type} agent stopped")
        
    async def _handle_message(self, message: AgentMessage):
        """Handle incoming messages"""
        try:
            self.messages_processed += 1
            response = await self.process_message(message)
            
            # Send response if correlation_id is present (request-response pattern)
            if message.correlation_id and response:
                response_msg = AgentMessage(
                    sender=self.agent_type,
                    recipient=message.sender,
                    message_type=MessageType.ANALYSIS_RESPONSE,
                    payload=response.dict(),
                    correlation_id=message.correlation_id
                )
                await self.message_bus.publish(response_msg)
                
        except Exception as e:
            self.errors_count += 1
            logger.error(f"Error processing message in {self.agent_type}: {e}")
            
            # Send error response
            if message.correlation_id:
                error_msg = AgentMessage(
                    sender=self.agent_type,
                    recipient=message.sender,
                    message_type=MessageType.ERROR,
                    payload={"error": str(e)},
                    correlation_id=message.correlation_id
                )
                await self.message_bus.publish(error_msg)
                
    async def send_message(self, 
                          recipient: Optional[AgentType],
                          message_type: MessageType,
                          payload: Dict[str, Any],
                          correlation_id: Optional[str] = None,
                          priority: int = 5):
        """Send a message via the message bus"""
        message = AgentMessage(
            sender=self.agent_type,
            recipient=recipient,
            message_type=message_type,
            payload=payload,
            correlation_id=correlation_id,
            priority=priority
        )
        await self.message_bus.publish(message)
        
    async def request_response(self,
                             recipient: AgentType,
                             message_type: MessageType,
                             payload: Dict[str, Any],
                             timeout: float = 30.0) -> Optional[AgentResponse]:
        """Send a request and wait for response"""
        correlation_id = str(uuid.uuid4())
        
        # Set up response handler
        response_future = asyncio.Future()
        
        async def response_handler(message: AgentMessage):
            if (message.correlation_id == correlation_id and 
                message.sender == recipient):
                if not response_future.done():
                    if message.message_type == MessageType.ERROR:
                        response_future.set_exception(Exception(message.payload.get("error", "Unknown error")))
                    else:
                        response_future.set_result(AgentResponse(**message.payload))
        
        # Temporarily subscribe for this specific response
        self.message_bus.subscribe(self.agent_type, response_handler)
        
        try:
            # Send request
            await self.send_message(
                recipient=recipient,
                message_type=message_type,
                payload=payload,
                correlation_id=correlation_id
            )
            
            # Wait for response
            response = await asyncio.wait_for(response_future, timeout=timeout)
            return response
            
        except asyncio.TimeoutError:
            logger.warning(f"Request to {recipient} timed out after {timeout}s")
            return None
        except Exception as e:
            logger.error(f"Error in request-response to {recipient}: {e}")
            return None
        finally:
            self.message_bus.unsubscribe(self.agent_type, response_handler)
            
    async def heartbeat(self):
        """Send heartbeat signal"""
        self.last_heartbeat = datetime.now()
        await self.send_message(
            recipient=None,  # Broadcast
            message_type=MessageType.HEARTBEAT,
            payload={
                "agent_id": self.agent_id,
                "timestamp": self.last_heartbeat.isoformat(),
                "status": "healthy" if self.is_running else "stopped",
                "messages_processed": self.messages_processed,
                "errors_count": self.errors_count
            }
        )
        
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "agent_type": self.agent_type.value,
            "agent_id": self.agent_id,
            "is_running": self.is_running,
            "messages_processed": self.messages_processed,
            "errors_count": self.errors_count,
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "config": self.config
        }
        
    @abstractmethod
    async def _initialize(self):
        """Initialize agent-specific resources"""
        pass
        
    @abstractmethod
    async def _cleanup(self):
        """Cleanup agent-specific resources"""
        pass
        
    @abstractmethod
    async def process_message(self, message: AgentMessage) -> Optional[AgentResponse]:
        """Process incoming messages - must be implemented by subclasses"""
        pass