#!/usr/bin/env python3
"""
Quick test script to verify MCP server is working
"""

import requests
import json

def test_mcp_server():
    """Test the MCP server analyze endpoint"""
    
    url = "http://localhost:8000/mcp"
    
    # Test request for Apple stock analysis
    request_data = {
        "jsonrpc": "2.0",
        "id": "test_1",
        "method": "analyze",
        "params": {
            "symbol": "AAPL",
            "timeframe": "1d",
            "include_technical": True,
            "include_fundamental": True,
            "include_sentiment": True
        }
    }
    
    try:
        print("🧪 Testing MCP Server...")
        print(f"URL: {url}")
        print(f"Request: {json.dumps(request_data, indent=2)}")
        
        response = requests.post(url, json=request_data, timeout=10)
        
        print(f"\n📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Response:")
            print(json.dumps(result, indent=2))
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

if __name__ == "__main__":
    success = test_mcp_server()
    if success:
        print("\n🎉 MCP Server is working correctly!")
    else:
        print("\n💥 MCP Server test failed!")