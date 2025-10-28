
"""
AuraTrade - AI-Enabled Autonomous Trading Platform
Main entry point for the trading system
"""

import asyncio
import sys
import os
import signal
import logging
from datetime import datetime
from typing import List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional

# Import platform components
from src.auratrade_platform import AuraTradePlatform
from src.config import load_config


# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('auratrade.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class AuraTradeMain:
    """Main application class for AuraTrade"""
    
    def __init__(self):
        self.platform = None
        self.config = None
        self.running = False
        
    async def initialize(self):
        """Initialize the application"""
        try:
            # Load configuration
            logger.info("Loading configuration...")
            self.config = load_config()
            logger.info(f"Configuration loaded successfully")
            logger.info(f"Environment: {self.config.environment}")
            logger.info(f"Trading enabled: {self.config.broker.broker_type}")
            
            # Create platform
            logger.info("Creating AuraTrade platform...")
            self.platform = AuraTradePlatform(self.config)
            
            # Initialize platform
            await self.platform.initialize()
            
            logger.info("AuraTrade application initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AuraTrade: {e}")
            raise
            
    async def start(self):
        """Start the trading platform"""
        if self.running:
            logger.warning("Platform is already running")
            return
            
        try:
            # Initialize if not done
            if not self.platform:
                await self.initialize()
                
            # Start platform
            logger.info("Starting AuraTrade platform...")
            await self.platform.start()
            
            self.running = True
            logger.info("AuraTrade platform started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start AuraTrade platform: {e}")
            raise
            
    async def stop(self):
        """Stop the trading platform"""
        if not self.running:
            return
            
        logger.info("Stopping AuraTrade platform...")
        self.running = False
        
        if self.platform:
            await self.platform.stop()
            
        logger.info("AuraTrade platform stopped")
        
    async def run_trading_session(self, symbols: List[str] = None, duration_minutes: int = None):
        """Run a trading session"""
        if not symbols:
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
            
        logger.info(f"Starting trading session for symbols: {symbols}")
        
        if duration_minutes:
            logger.info(f"Session will run for {duration_minutes} minutes")
            
        session_start = datetime.now()
        cycle_count = 0
        
        try:
            while self.running:
                cycle_count += 1
                cycle_start = datetime.now()
                
                logger.info(f"=== Trading Cycle {cycle_count} ===")
                
                # Check if duration limit reached
                if duration_minutes:
                    elapsed = (datetime.now() - session_start).total_seconds() / 60
                    if elapsed >= duration_minutes:
                        logger.info(f"Session duration ({duration_minutes} minutes) reached")
                        break
                
                # Run trading cycle
                try:
                    results = await self.platform.run_trading_cycle(symbols)
                    
                    if results['success']:
                        logger.info("Trading cycle completed successfully")
                        
                        # Log results summary
                        for symbol, result in results['results'].items():
                            signal_success = result['signal']['success']
                            execution_success = result['execution']['success']
                            
                            if signal_success and execution_success:
                                trading_decision = result['signal'].get('trading_decision', {})
                                action = trading_decision.get('action', 'UNKNOWN')
                                confidence = trading_decision.get('confidence', 0)
                                logger.info(f"{symbol}: {action} (confidence: {confidence:.2f})")
                            else:
                                logger.warning(f"{symbol}: Failed - Signal: {signal_success}, Execution: {execution_success}")
                                
                    else:
                        logger.error(f"Trading cycle failed: {results}")
                        
                except Exception as e:
                    logger.error(f"Error in trading cycle {cycle_count}: {e}")
                    
                # Wait before next cycle (1 minute)
                cycle_duration = (datetime.now() - cycle_start).total_seconds()
                wait_time = max(0, 60 - cycle_duration)
                
                if wait_time > 0:
                    logger.info(f"Waiting {wait_time:.1f} seconds before next cycle...")
                    await asyncio.sleep(wait_time)
                    
        except KeyboardInterrupt:
            logger.info("Trading session interrupted by user")
        except Exception as e:
            logger.error(f"Error in trading session: {e}")
        finally:
            session_duration = (datetime.now() - session_start).total_seconds() / 60
            logger.info(f"Trading session completed. Duration: {session_duration:.1f} minutes, Cycles: {cycle_count}")
            
    async def get_status(self):
        """Get platform status"""
        if not self.platform:
            return {
                'status': 'NOT_INITIALIZED',
                'message': 'Platform not initialized'
            }
            
        return await self.platform.get_system_status()
        
    async def demo_mode(self):
        """Run in demo mode with sample operations"""
        logger.info("=== AuraTrade Demo Mode ===")
        
        # Show configuration
        logger.info("Configuration:")
        config_dict = self.config.to_dict()
        for section, values in config_dict.items():
            if isinstance(values, dict):
                logger.info(f"  {section}:")
                for key, value in values.items():
                    logger.info(f"    {key}: {value}")
            else:
                logger.info(f"  {section}: {values}")
                
        # Wait a moment for agents to fully initialize
        await asyncio.sleep(5)
        
        # Show system status
        status = await self.get_status()
        logger.info(f"System Status: {status['status']}")
        logger.info(f"Trading Enabled: {status['trading_enabled']}")
        logger.info(f"Active Agents: {len([a for a in status['agents'].values() if a['status'] == 'HEALTHY'])}")
        
        # Request sample trading signals
        demo_symbols = ['AAPL', 'GOOGL', 'TSLA']
        
        for symbol in demo_symbols:
            logger.info(f"\n--- Requesting trading signal for {symbol} ---")
            
            try:
                signal_result = await self.platform.request_trading_signal(symbol)
                
                if signal_result['success']:
                    decision = signal_result['trading_decision']
                    logger.info(f"Signal: {decision['action']} (confidence: {decision['confidence']:.2f})")
                    logger.info(f"Reasoning: {decision.get('reasoning', 'No reasoning provided')}")
                    
                    # Show risk analysis if available
                    if 'risk_analysis' in decision:
                        risk = decision['risk_analysis']
                        logger.info(f"Risk Score: {risk.get('risk_score', 'N/A')}")
                        logger.info(f"Position Size: {risk.get('position_size', 'N/A')}")
                        
                else:
                    logger.warning(f"Failed to get signal for {symbol}: {signal_result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"Error getting signal for {symbol}: {e}")
                
            # Wait between requests
            await asyncio.sleep(2)
            
        logger.info("\n=== Demo completed ===")


async def main():
    """Main entry point"""
    app = AuraTradeMain()
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(app.stop())
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Parse command line arguments
        import argparse
        parser = argparse.ArgumentParser(description='AuraTrade - AI Autonomous Trading Platform')
        parser.add_argument('--mode', choices=['demo', 'trade', 'status'], default='demo',
                          help='Run mode: demo (show capabilities), trade (live trading), status (show status)')
        parser.add_argument('--symbols', nargs='+', default=['AAPL', 'GOOGL', 'MSFT', 'TSLA'],
                          help='Trading symbols (space-separated)')
        parser.add_argument('--duration', type=int, default=None,
                          help='Trading session duration in minutes')
        
        args = parser.parse_args()
        
        # Initialize and start platform
        logger.info("=== AuraTrade Starting ===")
        await app.start()
        
        # Run based on mode
        if args.mode == 'demo':
            await app.demo_mode()
        elif args.mode == 'trade':
            await app.run_trading_session(args.symbols, args.duration)
        elif args.mode == 'status':
            status = await app.get_status()
            logger.info(f"Platform Status: {status}")
            
        # Keep running until interrupted
        if args.mode == 'trade':
            logger.info("Press Ctrl+C to stop trading...")
            while app.running:
                await asyncio.sleep(1)
                
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)
    finally:
        await app.stop()
        logger.info("AuraTrade shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
