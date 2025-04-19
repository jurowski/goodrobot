"""
Main API module for GoodRobot.
This module sets up the FastAPI application and includes all API routes.
"""

import asyncio
import io
import json
import logging
import os
import wave
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, validator
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.websockets import WebSocketDisconnect
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from .routes import research
from .database.research_db import ResearchDatabase
from motor.motor_asyncio import AsyncIOMotorClient

from src.settings import Settings
from src.voice_recognition.speech_to_text import SpeechToText
from src.voice_recognition.wake_word import WakeWordDetector
from src.utils.logging import websocket_logger
from src.audio.audio_processor import AudioProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="GoodRobot API",
    description="API for GoodRobot voice AI assistant",
    version="0.1.0",
)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Enhanced CORS configuration
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the current directory
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent

# Set up templates with absolute path
templates = Jinja2Templates(directory=str(project_root / "src" / "api" / "templates"))

# Mount static files with absolute path
app.mount(
    "/static",
    StaticFiles(directory=str(project_root / "src" / "api" / "static")),
    name="static",
)

# Initialize settings
settings = Settings()

# Initialize wake word detector and speech-to-text
wake_word_detector = WakeWordDetector(settings)
speech_to_text = SpeechToText(settings)


# Add Pydantic model for site creation
class SiteCreate(BaseModel):
    name: str
    domain: str
    description: str
    port: int

    @validator('port')
    def validate_port(cls, v):
        if v < 1024 or v > 65535:
            raise ValueError('Port number must be between 1024 and 65535')
        return v

    @validator('domain')
    def validate_domain(cls, v):
        if not v or len(v) > 255:
            raise ValueError('Domain must be between 1 and 255 characters')
        return v


class RedirectMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Skip redirects for WebSocket connections
        if request.url.path.startswith("/ws/"):
            return await call_next(request)

        # Never redirect in development
        if os.getenv("ENVIRONMENT", "development") == "development":
            return await call_next(request)

        # Only enforce www redirects in production
        if (
            request.url.hostname == "goodrobot.cloud"
            and not request.url.hostname.startswith("www.")
        ):
            return RedirectResponse(
                url=str(request.url).replace("goodrobot.cloud", "www.goodrobot.cloud"),
                status_code=301,
            )

        return await call_next(request)


# Add redirect middleware
app.add_middleware(RedirectMiddleware)


@app.get("/")
async def root():
    return {"message": "Welcome to GoodRobot API"}


@app.get("/health")
async def health_check():
    """Check the health of the API and its dependencies."""
    try:
        # Check if wake word detector is initialized
        if wake_word_detector.porcupine is None:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "error": "Wake word detector not initialized",
                },
            )

        return {
            "status": "healthy",
            "wake_word_detector": "initialized",
            "version": "0.1.0",
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503, content={"status": "unhealthy", "error": str(e)}
        )


@app.get("/voice", response_class=HTMLResponse)
async def voice_interface(request: Request):
    """Serve the voice interface HTML page."""
    return templates.TemplateResponse("voice.html", {"request": request})


class CustomHTTPException(HTTPException):
    def __init__(self, status_code: int, detail: str, log_message: str = None):
        super().__init__(status_code=status_code, detail=detail)
        if log_message:
            logger.error(log_message)


@app.post("/voice")
@limiter.limit("10/minute")
async def process_voice(request: Request, audio: UploadFile = File(...), language: str = Form(...)):
    try:
        contents = await audio.read()
        if not contents:
            raise CustomHTTPException(
                status_code=400,
                detail="Empty audio file",
                log_message="Received empty audio file"
            )
            
        # Process audio data
        audio_array = np.frombuffer(contents, dtype=np.int16)
        wake_word_detected = wake_word_detector.process_audio_chunk(contents)
        
        if not wake_word_detected:
            return {"wake_word_detected": False}
            
        transcription = await speech_to_text.transcribe_audio(contents)
        return {
            "wake_word_detected": True,
            "transcription": transcription
        }
            
    except CustomHTTPException as e:
        raise e
    except Exception as e:
        raise CustomHTTPException(
            status_code=500,
            detail="Internal server error",
            log_message=f"Error processing voice: {str(e)}"
        )


class WebSocketLogHandler(logging.Handler):
    def __init__(self, websocket):
        super().__init__()
        self.websocket = websocket
        self.setLevel(logging.DEBUG)
        self.log_file = Path("logs/voice_session.log")
        self.log_file.parent.mkdir(exist_ok=True)

    def emit(self, record):
        try:
            log_entry = {
                "type": "log",
                "data": {
                    "timestamp": record.created,
                    "level": record.levelname,
                    "message": record.getMessage(),
                    "source": "server",
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno,
                },
            }

            # Write to file
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            # Send to WebSocket if connected
            if self.websocket.client_state.value == 1:  # WebSocketState.CONNECTED
                asyncio.create_task(self.websocket.send_json(log_entry))
        except Exception as e:
            print(f"Error in WebSocketLogHandler: {e}")


class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.audio_processors: Dict[str, AudioProcessor] = {}
        self.logger = websocket_logger
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.audio_processors[client_id] = AudioProcessor()
        self.logger.logger.info(f"Client {client_id} connected")
        
        # Add WebSocket handler for client-specific logging
        ws_handler = self.logger.add_websocket_handler(websocket, client_id)
        return ws_handler
    
    async def disconnect(self, client_id: str, ws_handler=None):
        if client_id in self.active_connections:
            if ws_handler:
                self.logger.remove_websocket_handler(ws_handler)
            
            # Cleanup resources
            del self.active_connections[client_id]
            if client_id in self.audio_processors:
                await self.audio_processors[client_id].cleanup()
                del self.audio_processors[client_id]
            
            self.logger.logger.info(f"Client {client_id} disconnected")
    
    async def process_json_message(self, message: str, client_id: str):
        try:
            data = json.loads(message)
            self.logger.logger.debug(f"Received JSON message from client {client_id}: {data}")
            
            # Process JSON message based on type
            message_type = data.get("type")
            if message_type == "config":
                await self._handle_config_message(data, client_id)
            elif message_type == "command":
                await self._handle_command_message(data, client_id)
            else:
                self.logger.logger.warning(f"Unknown message type from client {client_id}: {message_type}")
                
        except json.JSONDecodeError as e:
            self.logger.logger.error(f"Invalid JSON from client {client_id}: {e}")
            await self._send_error(client_id, "Invalid JSON format")
        except Exception as e:
            self.logger.logger.error(f"Error processing JSON message from client {client_id}: {e}")
            await self._send_error(client_id, "Internal server error")
    
    async def process_binary_message(self, message: bytes, client_id: str):
        try:
            if client_id not in self.audio_processors:
                raise ValueError("No audio processor found for client")
            
            processor = self.audio_processors[client_id]
            result = await processor.process_audio(message)
            
            await self.active_connections[client_id].send_json({
                "type": "audio_processed",
                "data": result
            })
            
        except Exception as e:
            self.logger.logger.error(f"Error processing binary message from client {client_id}: {e}")
            await self._send_error(client_id, "Error processing audio data")
    
    async def _handle_config_message(self, data: dict, client_id: str):
        try:
            if client_id in self.audio_processors:
                await self.audio_processors[client_id].configure(data.get("config", {}))
                self.logger.logger.info(f"Updated configuration for client {client_id}")
        except Exception as e:
            self.logger.logger.error(f"Error configuring audio processor for client {client_id}: {e}")
            await self._send_error(client_id, "Configuration error")
    
    async def _handle_command_message(self, data: dict, client_id: str):
        try:
            command = data.get("command")
            if command == "start":
                # Handle start command
                pass
            elif command == "stop":
                # Handle stop command
                pass
            else:
                self.logger.logger.warning(f"Unknown command from client {client_id}: {command}")
        except Exception as e:
            self.logger.logger.error(f"Error handling command for client {client_id}: {e}")
            await self._send_error(client_id, "Command processing error")
    
    async def _send_error(self, client_id: str, message: str):
        try:
            if client_id in self.active_connections:
                await self.active_connections[client_id].send_json({
                    "type": "error",
                    "message": message
                })
        except Exception as e:
            self.logger.logger.error(f"Error sending error message to client {client_id}: {e}")

websocket_manager = WebSocketManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    ws_handler = await websocket_manager.connect(websocket, client_id)
    
    try:
        while True:
            message = await websocket.receive()
            
            if message["type"] == "websocket.disconnect":
                break
                
            if message["type"] == "websocket.receive":
                if "text" in message:
                    await websocket_manager.process_json_message(message["text"], client_id)
                elif "bytes" in message:
                    await websocket_manager.process_binary_message(message["bytes"], client_id)
    
    except Exception as e:
        websocket_manager.logger.logger.error(f"WebSocket error for client {client_id}: {e}")
    
    finally:
        await websocket_manager.disconnect(client_id, ws_handler)


async def process_audio(audio_array: np.ndarray) -> dict:
    """Process audio data for wake word detection and transcription.

    Args:
        audio_array: Audio data as numpy array

    Returns:
        dict: Processing results
    """
    try:
        logger.debug(
            f"Processing audio array - shape: {audio_array.shape}, dtype: {audio_array.dtype}"
        )
        logger.debug(
            f"Audio array stats - min: {audio_array.min()}, max: {audio_array.max()}, mean: {audio_array.mean()}"
        )

        # Convert to bytes for wake word detection
        audio_bytes = audio_array.tobytes()
        logger.debug(f"Converted to bytes - length: {len(audio_bytes)}")

        # Check for wake word
        logger.debug("Starting wake word detection")
        wake_word_detected = wake_word_detector.process_audio_chunk(audio_bytes)
        logger.debug(f"Wake word detection result: {wake_word_detected}")

        if wake_word_detected:
            logger.info("Wake word detected, starting transcription")
            # If wake word detected, also transcribe the audio
            transcription = await speech_to_text.transcribe_audio(audio_bytes)
            logger.info(f"Transcription completed: {transcription}")
            return {
                "type": "result",
                "wake_word_detected": True,
                "transcription": transcription,
            }

        logger.debug("No wake word detected")
        return {"type": "result", "wake_word_detected": False}

    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error traceback: {e.__traceback__}")
        return {"type": "error", "error": str(e)}


@app.post("/api/sites")
async def create_site(site: SiteCreate):
    """Create a new site."""
    try:
        # Validate the port number
        if site.port < 1024 or site.port > 65535:
            raise HTTPException(
                status_code=400, detail="Port number must be between 1024 and 65535"
            )

        # TODO: Add your site creation logic here
        # For now, just return the created site data
        return JSONResponse(
            status_code=201,
            content={
                "status": "success",
                "message": "Site created successfully",
                "data": site.dict(),
            },
        )
    except Exception as e:
        logger.error(f"Error creating site: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


# Ensure required directories exist
def ensure_directories():
    """Create required directories if they don't exist."""
    directories = [
        project_root / "src" / "api" / "templates",
        project_root / "src" / "api" / "static",
        project_root / "src" / "api" / "static" / "images",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory exists: {directory.exists()}")


# Initialize directories
ensure_directories()

app.include_router(research.router)

@app.get("/research")
async def research_page(request: Request):
    """Render the research page."""
    return templates.TemplateResponse(
        "research.html",
        {"request": request}
    )

# Add MongoDB client to app state
@app.on_event("startup")
async def startup_db_client():
    app.mongodb_client = AsyncIOMotorClient("mongodb://localhost:27017")
    app.research_db = ResearchDatabase(app.mongodb_client)

@app.on_event("shutdown")
async def shutdown_db_client():
    app.mongodb_client.close()
