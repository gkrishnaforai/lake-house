"""WebSocket manager for real-time classification updates."""

from fastapi import WebSocket
from typing import Dict, Optional
import json
import logging
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class ClassificationProgress(BaseModel):
    """Progress update for classification process."""
    status: str
    progress: float
    current_operation: str
    error: Optional[str] = None

class ClassificationPreview(BaseModel):
    """Preview of classification results."""
    sample_data: list
    explanation: str
    metadata: dict

class WebSocketManager:
    """Manager for WebSocket connections and message broadcasting."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str):
        """Connect a new client."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")
        
    def disconnect(self, client_id: str):
        """Disconnect a client."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected")
            
    async def broadcast_progress(self, client_id: str, progress: ClassificationProgress):
        """Send progress update to a specific client."""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(progress.dict())
            except Exception as e:
                logger.error(f"Error sending progress to client {client_id}: {str(e)}")
                self.disconnect(client_id)
                
    async def broadcast_preview(self, client_id: str, preview: ClassificationPreview):
        """Send classification preview to a specific client."""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(preview.dict())
            except Exception as e:
                logger.error(f"Error sending preview to client {client_id}: {str(e)}")
                self.disconnect(client_id)
                
    async def broadcast_error(self, client_id: str, error: str):
        """Send error message to a specific client."""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json({
                    "type": "error",
                    "message": error
                })
            except Exception as e:
                logger.error(f"Error sending error to client {client_id}: {str(e)}")
                self.disconnect(client_id)

# Create a global WebSocket manager instance
websocket_manager = WebSocketManager() 