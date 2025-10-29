#!/usr/bin/env python3
"""
Debug script to test intent extraction
"""

import re
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from chatbot.trading_chatbot import SimpleNLPProcessor

def test_intent_extraction():
    """Test intent extraction with the problematic query"""
    
    nlp = SimpleNLPProcessor()
    
    test_queries = [
        "Analyze Apple stock",
        "Analyze AAPL",
        "What about Apple?",
        "Tell me about AAPL",
        "Apple analysis"
    ]
    
    for query in test_queries:
        print(f"\n🧪 Testing: '{query}'")
        intent = nlp.extract_intent(query)
        print(f"Intent: {intent.name}")
        print(f"Confidence: {intent.confidence}")
        print(f"Entities: {intent.entities}")
        
        # Check what the pattern matching finds
        text_lower = query.lower()
        for intent_name, patterns in nlp.intent_patterns.items():
            if intent_name == "market_analysis":
                print(f"\n  Market analysis patterns:")
                for pattern in patterns:
                    match = re.search(pattern, text_lower)
                    if match:
                        print(f"    ✅ Matched pattern: '{pattern}' -> groups: {match.groups()}")
                    else:
                        print(f"    ❌ No match: '{pattern}'")

if __name__ == "__main__":
    test_intent_extraction()