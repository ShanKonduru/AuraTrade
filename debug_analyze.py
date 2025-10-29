#!/usr/bin/env python3
"""
Debug script to trace the exact error in "Analyze Apple stock" request
"""

import asyncio
import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from chatbot.trading_chatbot import TradingChatbot

async def debug_analyze_apple():
    """Debug the exact 'Analyze Apple stock' request"""
    
    try:
        # Create chatbot instance
        chatbot = TradingChatbot("http://localhost:8000")
        
        # Test the exact message
        user_input = "Analyze Apple stock"
        
        print(f"🧪 Testing: '{user_input}'")
        print("=" * 50)
        
        # Process the message step by step
        print("\n1. Extracting intent...")
        intent = chatbot.nlp.extract_intent(user_input)
        print(f"Intent: {intent.name}")
        print(f"Confidence: {intent.confidence}")
        print(f"Entities: {intent.entities}")
        
        print("\n2. Processing message...")
        response = await chatbot.process_message(user_input)
        print(f"Response: {response}")
        
        print("\n✅ Success!")
        
    except Exception as e:
        print(f"\n❌ Error occurred:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"\nFull traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_analyze_apple())