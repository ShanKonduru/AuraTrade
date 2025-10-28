#!/usr/bin/env python3
"""
AuraTrade Demo Launcher
Quick start script for testing MCP server and chatbot integration
"""

import asyncio
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
import requests
import threading

def check_port(port):
    """Check if a port is available"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0

def wait_for_server(url, timeout=30):
    """Wait for server to be ready"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    return False

def run_mcp_server(port=8000):
    """Run MCP server in background"""
    print(f"🚀 Starting MCP Server on port {port}...")
    
    # Change to project directory
    project_dir = Path(__file__).parent
    
    # Start MCP server
    process = subprocess.Popen([
        sys.executable, "-m", "uvicorn", 
        "src.mcp_server.server:app",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--reload"
    ], cwd=project_dir)
    
    return process

def run_chatbot_interface(port=8080):
    """Run chatbot web interface"""
    print(f"🤖 Starting Chatbot Interface on port {port}...")
    
    # Change to project directory  
    project_dir = Path(__file__).parent
    
    # Start chatbot interface
    process = subprocess.Popen([
        sys.executable, "-m", "uvicorn",
        "src.chatbot.chat_interface:app", 
        "--host", "0.0.0.0",
        "--port", str(port),
        "--reload"
    ], cwd=project_dir)
    
    return process

def main():
    """Main launcher function"""
    print("🌟 AuraTrade AI Trading Platform Launcher")
    print("=" * 50)
    
    # Check if ports are available
    if not check_port(8000):
        print("❌ Port 8000 is already in use. Trying to use port 8001 instead.")
        mcp_port = 8001
    else:
        mcp_port = 8000
    
    if not check_port(8080):
        print("❌ Port 8080 is already in use. Trying to use port 8081 instead.")
        chat_port = 8081
    else:
        chat_port = 8080
    
    try:
        # Start MCP server
        mcp_process = run_mcp_server(mcp_port)
        time.sleep(3)  # Give server time to start
        
        # Wait for MCP server to be ready
        if wait_for_server(f"http://localhost:{mcp_port}/health"):
            print(f"✅ MCP Server is running at http://localhost:{mcp_port}")
        else:
            print("❌ MCP Server failed to start")
            mcp_process.terminate()
            return
        
        # Start chatbot interface
        chat_process = run_chatbot_interface(chat_port)
        time.sleep(3)  # Give interface time to start
        
        # Wait for chatbot interface to be ready
        if wait_for_server(f"http://localhost:{chat_port}/health"):
            print(f"✅ Chatbot Interface is running at http://localhost:{chat_port}")
        else:
            print("❌ Chatbot Interface failed to start")
            mcp_process.terminate()
            chat_process.terminate()
            return
        
        print("\n" + "🎉 AuraTrade Platform is Ready!" + "\n")
        print(f"📊 MCP Server API: http://localhost:{mcp_port}")
        print(f"📊 API Documentation: http://localhost:{mcp_port}/docs")
        print(f"🤖 Chatbot Interface: http://localhost:{chat_port}")
        
        # Auto-open browser
        try:
            webbrowser.open(f"http://localhost:{chat_port}")
            print("\n🌐 Opening chatbot interface in your browser...")
        except:
            pass
        
        print("\n💡 Usage:")
        print(f"• Visit http://localhost:{chat_port} for the trading chatbot")
        print("• Ask questions like 'Analyze Apple stock' or 'Show me growth stocks'")
        print(f"• Use the API at http://localhost:{mcp_port} for programmatic access")
        print("\n🛑 Press Ctrl+C to stop all services")
        
        # Keep running until interrupted
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n⏹️  Stopping AuraTrade Platform...")
            mcp_process.terminate()
            chat_process.terminate()
            
            # Wait for processes to finish
            mcp_process.wait(timeout=5)
            chat_process.wait(timeout=5)
            
            print("✅ All services stopped successfully")
            
    except Exception as e:
        print(f"❌ Error launching platform: {e}")
        return

if __name__ == "__main__":
    main()