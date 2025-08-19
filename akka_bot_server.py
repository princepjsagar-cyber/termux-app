import asyncio
import json
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Any
from aiohttp import web
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Define immutable state model
@dataclass
class BotState:
    history: List[str] = field(default_factory=list)
    user_data: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, text: str) -> 'BotState':
        """Create new state with added message (immutable)"""
        new_history = [text] + self.history
        return BotState(history=new_history, user_data=self.user_data.copy())
    
    def update_user_data(self, user_id: str, data: Any) -> 'BotState':
        """Create new state with updated user data (immutable)"""
        new_user_data = self.user_data.copy()
        new_user_data[user_id] = data
        return BotState(history=self.history, user_data=new_user_data)

# 2. Actor-like state manager
class BotAgent:
    def __init__(self, initial_state: BotState):
        self._state = initial_state
        self._lock = threading.Lock()
    
    def process_message(self, text: str) -> str:
        """Process message and return response (thread-safe)"""
        with self._lock:
            self._state = self._state.add_message(text)
            return f"Processed: '{text}' | History size: {len(self._state.history)}"
    
    def get_state(self) -> BotState:
        """Get current state (thread-safe)"""
        with self._lock:
            return self._state
    
    def update_user_data(self, user_id: str, data: Any) -> str:
        """Update user data (thread-safe)"""
        with self._lock:
            self._state = self._state.update_user_data(user_id, data)
            return f"Updated user {user_id} data"

# 3. HTTP server to keep bot live
class BotServer:
    def __init__(self):
        # Initialize with existing data (replace with your actual data)
        existing_data = BotState(
            history=["Old message 1", "Old message 2"],
            user_data={"user1": {"items": ["existing item 1", "existing item 2"]}}
        )
        self.bot_agent = BotAgent(existing_data)
        self.app = web.Application()
        self.setup_routes()
    
    def setup_routes(self):
        """Setup HTTP routes"""
        self.app.router.add_post('/message/{text}', self.handle_message)
        self.app.router.add_get('/state', self.handle_get_state)
        self.app.router.add_post('/user/{user_id}/data', self.handle_user_data)
        self.app.router.add_get('/health', self.handle_health)
    
    async def handle_message(self, request):
        """Handle message processing"""
        text = request.match_info['text']
        try:
            response = self.bot_agent.process_message(text)
            return web.json_response({
                "status": "success",
                "response": response,
                "timestamp": asyncio.get_event_loop().time()
            })
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return web.json_response({
                "status": "error",
                "error": str(e)
            }, status=500)
    
    async def handle_get_state(self, request):
        """Handle state retrieval"""
        try:
            state = self.bot_agent.get_state()
            return web.json_response({
                "status": "success",
                "state": {
                    "history": state.history,
                    "history_size": len(state.history),
                    "user_data_keys": list(state.user_data.keys())
                }
            })
        except Exception as e:
            logger.error(f"Error getting state: {e}")
            return web.json_response({
                "status": "error",
                "error": str(e)
            }, status=500)
    
    async def handle_user_data(self, request):
        """Handle user data updates"""
        user_id = request.match_info['user_id']
        try:
            data = await request.json()
            response = self.bot_agent.update_user_data(user_id, data)
            return web.json_response({
                "status": "success",
                "response": response
            })
        except Exception as e:
            logger.error(f"Error updating user data: {e}")
            return web.json_response({
                "status": "error",
                "error": str(e)
            }, status=500)
    
    async def handle_health(self, request):
        """Health check endpoint"""
        return web.json_response({
            "status": "healthy",
            "service": "akka-bot-server",
            "timestamp": asyncio.get_event_loop().time()
        })
    
    async def start(self, host='0.0.0.0', port=8070):
        """Start the HTTP server"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        logger.info(f"Bot server running at http://{host}:{port}")
        logger.info("Available endpoints:")
        logger.info("  POST /message/{text} - Process a message")
        logger.info("  GET  /state - Get current state")
        logger.info("  POST /user/{user_id}/data - Update user data")
        logger.info("  GET  /health - Health check")
        
        # Keep the server running
        try:
            await asyncio.Future()  # Run forever
        except KeyboardInterrupt:
            logger.info("Shutting down server...")
        finally:
            await runner.cleanup()

async def main():
    """Main entry point"""
    server = BotServer()
    await server.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")