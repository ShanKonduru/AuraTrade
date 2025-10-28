#!/usr/bin/env python3
"""
AuraTrade Real Data Demo
This version fetches actual real-time market data
"""

import sys
import asyncio
import yfinance as yf
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def print_banner():
    """Print banner"""
    banner = """
    ==============================================
    AuraTrade - REAL DATA Demo
    ==============================================
    
    Fetching Live Market Data from Yahoo Finance
    
    """
    print(banner)

def get_real_market_data(symbols):
    """Fetch real market data"""
    print("📊 Fetching REAL market data...")
    
    data = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            
            # Get current price
            hist = ticker.history(period="1d", interval="1m")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                
                # Get basic info
                info = ticker.info
                
                data[symbol] = {
                    'price': current_price,
                    'change': hist['Close'].iloc[-1] - hist['Open'].iloc[0],
                    'volume': hist['Volume'].iloc[-1] if 'Volume' in hist else 0,
                    'market_cap': info.get('marketCap', 'N/A'),
                    'pe_ratio': info.get('trailingPE', 'N/A'),
                    'day_high': hist['High'].max(),
                    'day_low': hist['Low'].min()
                }
                
                print(f"  ✅ {symbol}: ${current_price:.2f}")
            else:
                print(f"  ❌ {symbol}: No data available")
                
        except Exception as e:
            print(f"  ❌ {symbol}: Error - {str(e)}")
            
    return data

def calculate_real_technical_indicators(symbol):
    """Calculate real technical indicators"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="30d", interval="1d")
        
        if len(hist) < 14:
            return None
            
        # Simple RSI calculation
        delta = hist['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Simple moving averages
        sma_20 = hist['Close'].rolling(window=20).mean()
        sma_50 = hist['Close'].rolling(window=min(50, len(hist))).mean()
        
        current_price = hist['Close'].iloc[-1]
        
        return {
            'rsi': rsi.iloc[-1] if not rsi.empty else None,
            'sma_20': sma_20.iloc[-1] if not sma_20.empty else None,
            'sma_50': sma_50.iloc[-1] if not sma_50.empty else None,
            'current_price': current_price,
            'trend': 'Bullish' if current_price > sma_20.iloc[-1] else 'Bearish' if not pd.isna(sma_20.iloc[-1]) else 'Neutral'
        }
        
    except Exception as e:
        print(f"Error calculating indicators for {symbol}: {e}")
        return None

async def real_data_demo():
    """Demo with real market data"""
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    
    print_banner()
    
    # Get real market data
    market_data = get_real_market_data(symbols)
    
    if not market_data:
        print("❌ Unable to fetch real market data. Check internet connection.")
        return
    
    print(f"\n📊 Real Market Data (as of {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}):")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    for symbol, data in market_data.items():
        change_sign = "+" if data['change'] >= 0 else ""
        print(f"📈 {symbol}:")
        print(f"   💰 Price: ${data['price']:.2f} ({change_sign}{data['change']:.2f})")
        print(f"   📊 Volume: {data['volume']:,}")
        print(f"   📈 Day Range: ${data['day_low']:.2f} - ${data['day_high']:.2f}")
        if data['pe_ratio'] != 'N/A':
            print(f"   📋 P/E Ratio: {data['pe_ratio']:.2f}")
        print()
    
    # Calculate real technical indicators
    print("🔍 Real Technical Analysis:")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    for symbol in symbols:
        print(f"📈 {symbol} Technical Indicators:")
        indicators = calculate_real_technical_indicators(symbol)
        
        if indicators:
            rsi = indicators['rsi']
            sma_20 = indicators['sma_20']
            trend = indicators['trend']
            current_price = indicators['current_price']
            
            if rsi:
                rsi_signal = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"
                print(f"   📊 RSI: {rsi:.1f} ({rsi_signal})")
            
            if sma_20:
                print(f"   📈 20-day SMA: ${sma_20:.2f}")
                position = "Above" if current_price > sma_20 else "Below"
                print(f"   🎯 Price vs SMA: {position} ({trend})")
            
            # Generate real signal
            if rsi and rsi < 35 and trend == 'Bullish':
                signal = "STRONG BUY"
                confidence = "90%"
            elif rsi and rsi < 40 and trend == 'Bullish':
                signal = "BUY"
                confidence = "75%"
            elif rsi and rsi > 65 and trend == 'Bearish':
                signal = "SELL"
                confidence = "70%"
            else:
                signal = "HOLD"
                confidence = "50%"
                
            print(f"   🎯 Signal: {signal} (Confidence: {confidence})")
        else:
            print(f"   ❌ Unable to calculate indicators")
        
        print()
    
    print("🎉 Real data analysis complete!")
    print("\n📝 Note: This is REAL market data from Yahoo Finance")
    print("🎯 Signals are based on actual RSI and moving average calculations")
    print("💡 For full system with LLM analysis, see setup guide")

if __name__ == "__main__":
    try:
        asyncio.run(real_data_demo())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("💡 Make sure you have internet connection for real data")