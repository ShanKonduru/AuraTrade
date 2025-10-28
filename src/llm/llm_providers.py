"""
LLM Provider abstraction for flexible LLM integration
Supports OpenAI, Ollama, and other LLM providers
"""

import asyncio
import json
import aiohttp
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"


@dataclass
class LLMResponse:
    """Standardized LLM response"""
    content: str
    success: bool
    error: Optional[str] = None
    model: Optional[str] = None
    tokens_used: Optional[int] = None
    cost: Optional[float] = None


@dataclass
class LLMConfig:
    """LLM configuration"""
    provider: LLMProvider
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: int = 30


class BaseLLMProvider(ABC):
    """Base class for LLM providers"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        
    @abstractmethod
    async def generate_text(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate text completion"""
        pass
        
    @abstractmethod
    async def generate_chat(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Generate chat completion"""
        pass
        
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available"""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import openai
            self.client = openai.AsyncOpenAI(api_key=config.api_key)
            self.available = True
        except ImportError:
            self.available = False
            
    def is_available(self) -> bool:
        return self.available and bool(self.config.api_key)
        
    async def generate_text(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate text using OpenAI"""
        if not self.is_available():
            return LLMResponse(
                content="",
                success=False,
                error="OpenAI not available or API key missing"
            )
            
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                success=True,
                model=self.config.model,
                tokens_used=response.usage.total_tokens if response.usage else None
            )
            
        except Exception as e:
            return LLMResponse(
                content="",
                success=False,
                error=f"OpenAI error: {str(e)}"
            )
            
    async def generate_chat(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Generate chat completion using OpenAI"""
        if not self.is_available():
            return LLMResponse(
                content="",
                success=False,
                error="OpenAI not available or API key missing"
            )
            
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                success=True,
                model=self.config.model,
                tokens_used=response.usage.total_tokens if response.usage else None
            )
            
        except Exception as e:
            return LLMResponse(
                content="",
                success=False,
                error=f"OpenAI error: {str(e)}"
            )


class OllamaProvider(BaseLLMProvider):
    """Ollama local LLM provider"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.base_url or "http://localhost:11434"
        
    def is_available(self) -> bool:
        """Check if Ollama is running"""
        try:
            import asyncio
            import aiohttp
            # This will be checked properly in async context
            return True
        except ImportError:
            return False
            
    async def _check_ollama_health(self) -> bool:
        """Check if Ollama service is running"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags", timeout=5) as response:
                    return response.status == 200
        except:
            return False
            
    async def generate_text(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate text using Ollama"""
        if not await self._check_ollama_health():
            return LLMResponse(
                content="",
                success=False,
                error="Ollama service not available. Please ensure Ollama is running."
            )
            
        try:
            # Prepare the full prompt
            full_prompt = ""
            if system_prompt:
                full_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            else:
                full_prompt = prompt
                
            payload = {
                "model": self.config.model,
                "prompt": full_prompt,
                "temperature": self.config.temperature,
                "stream": False
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.config.timeout
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return LLMResponse(
                            content=result.get("response", ""),
                            success=True,
                            model=self.config.model
                        )
                    else:
                        error_text = await response.text()
                        return LLMResponse(
                            content="",
                            success=False,
                            error=f"Ollama API error: {response.status} - {error_text}"
                        )
                        
        except Exception as e:
            return LLMResponse(
                content="",
                success=False,
                error=f"Ollama error: {str(e)}"
            )
            
    async def generate_chat(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Generate chat completion using Ollama"""
        if not await self._check_ollama_health():
            return LLMResponse(
                content="",
                success=False,
                error="Ollama service not available. Please ensure Ollama is running."
            )
            
        try:
            payload = {
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.temperature,
                "stream": False
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=self.config.timeout
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result.get("message", {}).get("content", "")
                        return LLMResponse(
                            content=content,
                            success=True,
                            model=self.config.model
                        )
                    else:
                        error_text = await response.text()
                        return LLMResponse(
                            content="",
                            success=False,
                            error=f"Ollama API error: {response.status} - {error_text}"
                        )
                        
        except Exception as e:
            return LLMResponse(
                content="",
                success=False,
                error=f"Ollama error: {str(e)}"
            )


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude LLM provider"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=config.api_key)
            self.available = True
        except ImportError:
            self.available = False
            
    def is_available(self) -> bool:
        return self.available and bool(self.config.api_key)
        
    async def generate_text(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate text using Anthropic Claude"""
        if not self.is_available():
            return LLMResponse(
                content="",
                success=False,
                error="Anthropic not available or API key missing"
            )
            
        try:
            messages = [{"role": "user", "content": prompt}]
            
            kwargs = {
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens
            }
            
            if system_prompt:
                kwargs["system"] = system_prompt
                
            response = await self.client.messages.create(**kwargs)
            
            return LLMResponse(
                content=response.content[0].text,
                success=True,
                model=self.config.model,
                tokens_used=response.usage.input_tokens + response.usage.output_tokens
            )
            
        except Exception as e:
            return LLMResponse(
                content="",
                success=False,
                error=f"Anthropic error: {str(e)}"
            )
            
    async def generate_chat(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Generate chat completion using Anthropic Claude"""
        if not self.is_available():
            return LLMResponse(
                content="",
                success=False,
                error="Anthropic not available or API key missing"
            )
            
        try:
            # Extract system message if present
            system_prompt = None
            filtered_messages = []
            
            for msg in messages:
                if msg.get("role") == "system":
                    system_prompt = msg.get("content")
                else:
                    filtered_messages.append(msg)
                    
            kwargs = {
                "model": self.config.model,
                "messages": filtered_messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens
            }
            
            if system_prompt:
                kwargs["system"] = system_prompt
                
            response = await self.client.messages.create(**kwargs)
            
            return LLMResponse(
                content=response.content[0].text,
                success=True,
                model=self.config.model,
                tokens_used=response.usage.input_tokens + response.usage.output_tokens
            )
            
        except Exception as e:
            return LLMResponse(
                content="",
                success=False,
                error=f"Anthropic error: {str(e)}"
            )


class LLMManager:
    """Manager for multiple LLM providers with fallback support"""
    
    def __init__(self):
        self.providers: Dict[LLMProvider, BaseLLMProvider] = {}
        self.primary_provider: Optional[LLMProvider] = None
        self.fallback_providers: List[LLMProvider] = []
        
    def add_provider(self, provider_type: LLMProvider, config: LLMConfig, 
                    is_primary: bool = False, is_fallback: bool = True):
        """Add an LLM provider"""
        if provider_type == LLMProvider.OPENAI:
            provider = OpenAIProvider(config)
        elif provider_type == LLMProvider.OLLAMA:
            provider = OllamaProvider(config)
        elif provider_type == LLMProvider.ANTHROPIC:
            provider = AnthropicProvider(config)
        else:
            raise ValueError(f"Unsupported provider: {provider_type}")
            
        self.providers[provider_type] = provider
        
        if is_primary:
            self.primary_provider = provider_type
            
        if is_fallback and provider_type not in self.fallback_providers:
            self.fallback_providers.append(provider_type)
            
    async def generate_text(self, prompt: str, system_prompt: Optional[str] = None,
                          preferred_provider: Optional[LLMProvider] = None) -> LLMResponse:
        """Generate text with automatic fallback"""
        
        # Determine provider order
        providers_to_try = []
        
        if preferred_provider and preferred_provider in self.providers:
            providers_to_try.append(preferred_provider)
            
        if self.primary_provider and self.primary_provider not in providers_to_try:
            providers_to_try.append(self.primary_provider)
            
        # Add fallback providers
        for provider in self.fallback_providers:
            if provider not in providers_to_try:
                providers_to_try.append(provider)
                
        # Try providers in order
        last_error = "No providers available"
        
        for provider_type in providers_to_try:
            provider = self.providers.get(provider_type)
            if provider and provider.is_available():
                try:
                    response = await provider.generate_text(prompt, system_prompt)
                    if response.success:
                        return response
                    else:
                        last_error = response.error or f"{provider_type.value} failed"
                except Exception as e:
                    last_error = f"{provider_type.value} error: {str(e)}"
                    
        return LLMResponse(
            content="",
            success=False,
            error=f"All providers failed. Last error: {last_error}"
        )
        
    async def generate_chat(self, messages: List[Dict[str, str]],
                          preferred_provider: Optional[LLMProvider] = None) -> LLMResponse:
        """Generate chat completion with automatic fallback"""
        
        # Determine provider order
        providers_to_try = []
        
        if preferred_provider and preferred_provider in self.providers:
            providers_to_try.append(preferred_provider)
            
        if self.primary_provider and self.primary_provider not in providers_to_try:
            providers_to_try.append(self.primary_provider)
            
        # Add fallback providers
        for provider in self.fallback_providers:
            if provider not in providers_to_try:
                providers_to_try.append(provider)
                
        # Try providers in order
        last_error = "No providers available"
        
        for provider_type in providers_to_try:
            provider = self.providers.get(provider_type)
            if provider and provider.is_available():
                try:
                    response = await provider.generate_chat(messages)
                    if response.success:
                        return response
                    else:
                        last_error = response.error or f"{provider_type.value} failed"
                except Exception as e:
                    last_error = f"{provider_type.value} error: {str(e)}"
                    
        return LLMResponse(
            content="",
            success=False,
            error=f"All providers failed. Last error: {last_error}"
        )
        
    def get_available_providers(self) -> List[LLMProvider]:
        """Get list of available providers"""
        available = []
        for provider_type, provider in self.providers.items():
            if provider.is_available():
                available.append(provider_type)
        return available
        
    def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all providers"""
        status = {}
        for provider_type, provider in self.providers.items():
            status[provider_type.value] = {
                "available": provider.is_available(),
                "model": provider.config.model,
                "is_primary": provider_type == self.primary_provider,
                "is_fallback": provider_type in self.fallback_providers
            }
        return status


# Factory function for easy LLM manager creation
def create_llm_manager(config_dict: Dict[str, Any]) -> LLMManager:
    """Create LLM manager from configuration dictionary"""
    manager = LLMManager()
    
    # Parse provider configurations
    for provider_name, provider_config in config_dict.items():
        if not provider_config.get("enabled", False):
            continue
            
        provider_type = LLMProvider(provider_name.lower())
        
        llm_config = LLMConfig(
            provider=provider_type,
            model=provider_config.get("model", ""),
            api_key=provider_config.get("api_key"),
            base_url=provider_config.get("base_url"),
            temperature=provider_config.get("temperature", 0.7),
            max_tokens=provider_config.get("max_tokens", 1000),
            timeout=provider_config.get("timeout", 30)
        )
        
        manager.add_provider(
            provider_type,
            llm_config,
            is_primary=provider_config.get("is_primary", False),
            is_fallback=provider_config.get("is_fallback", True)
        )
        
    return manager