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

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.websockets import WebSocketDisconnect

from src.settings import Settings
from src.voice_recognition.speech_to_text import SpeechToText
from src.voice_recognition.wake_word import WakeWordDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="GoodRobot API",
    description="API for GoodRobot voice AI assistant",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
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


@app.post("/voice")
async def process_voice(audio: UploadFile = File(...), language: str = Form(...)):
    """Process voice input and return response."""
    try:
        # Read audio file
        audio_data = await audio.read()

        # Check for wake word
        if wake_word_detector.process_audio_chunk(audio_data):
            # Transcribe audio
            transcription = await speech_to_text.transcribe_audio(audio_data)

            # TODO: Process the transcription and generate response
            response = f"Transcription: {transcription}"

            return {"transcription": transcription, "response": response}
        else:
            return {
                "transcription": "No wake word detected",
                "response": "Please say the wake word first",
            }
    except Exception as e:
        logger.error(f"Error processing voice: {str(e)}")
        return {"error": str(e)}


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


@app.websocket("/voice")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Add WebSocket log handler
    ws_log_handler = WebSocketLogHandler(websocket)
    logger.addHandler(ws_log_handler)

    # Send recent logs from file
    try:
        log_file = Path("logs/voice_session.log")
        if log_file.exists():
            with open(log_file, "r") as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        if log_entry.get("type") == "log":
                            await websocket.send_json(log_entry)
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        logger.error(f"Error sending recent logs: {e}")

    logger.info("WebSocket connection opened")
    logger.info(f"WebSocket client: {websocket.client}")
    logger.info(f"WebSocket headers: {websocket.headers}")

    try:
        while True:
            try:
                message = await websocket.receive()
                logger.debug(f"Received message type: {message.get('type')}")
                logger.debug(f"Message content: {message}")

                if message["type"] == "websocket.disconnect":
                    logger.info("WebSocket disconnect message received")
                    logger.info(f"Disconnect code: {message.get('code')}")
                    logger.info(f"Disconnect reason: {message.get('reason')}")
                    break

                if message["type"] == "websocket.receive":
                    if "text" in message:
                        # Handle JSON messages
                        try:
                            data = json.loads(message["text"])
                            logger.debug(f"Received JSON data: {data}")
                            if data.get("type") == "config":
                                logger.info(f"Received config: {data.get('data')}")
                                await websocket.send_json({"type": "config_received"})
                                logger.debug("Sent config_received response")
                                continue
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to decode JSON message: {e}")
                            logger.error(f"Raw message text: {message.get('text')}")
                            continue

                    elif "bytes" in message:
                        # Handle binary audio data
                        try:
                            data = message["bytes"]
                            logger.debug(f"Received audio data size: {len(data)} bytes")
                            audio_array = np.frombuffer(data, dtype=np.int16)
                            logger.debug(
                                f"Audio array shape: {audio_array.shape}, dtype: {audio_array.dtype}"
                            )

                            # Process audio without reshaping assumptions
                            result = await process_audio(audio_array)
                            logger.debug(f"Processing result: {result}")

                            # Send result back
                            await websocket.send_json(result)
                            logger.debug("Sent processing result back to client")

                        except Exception as e:
                            logger.error(f"Error processing audio data: {str(e)}")
                            logger.error(
                                f"Audio data type: {type(data)}, length: {len(data) if hasattr(data, '__len__') else 'N/A'}"
                            )
                            await websocket.send_json(
                                {
                                    "type": "error",
                                    "error": "Failed to process audio data",
                                }
                            )
                            continue

            except WebSocketDisconnect as e:
                logger.info(f"WebSocket disconnected by client: {str(e)}")
                logger.info(f"Disconnect code: {getattr(e, 'code', 'N/A')}")
                logger.info(f"Disconnect reason: {getattr(e, 'reason', 'N/A')}")
                break
            except Exception as e:
                logger.error(f"Unexpected error in WebSocket loop: {str(e)}")
                logger.error(f"Error type: {type(e)}")
                logger.error(f"Error traceback: {e.__traceback__}")
                try:
                    await websocket.send_json(
                        {"type": "error", "error": "Internal server error"}
                    )
                except Exception as send_error:
                    logger.error(f"Failed to send error message: {str(send_error)}")
                    break

    except Exception as e:
        logger.error(f"WebSocket connection error: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error traceback: {e.__traceback__}")
    finally:
        logger.info("WebSocket connection closed")
        logger.info(f"WebSocket state: {websocket.client_state}")
        # Remove the WebSocket log handler
        logger.removeHandler(ws_log_handler)


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
