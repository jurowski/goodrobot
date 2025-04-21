"""
Main API module for GoodRobot.
This module sets up the FastAPI application and includes all API routes.
"""

import asyncio
import io
import json
import logging
import os
import traceback
import uuid
import wave
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf
from scipy import signal
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.websockets import WebSocketDisconnect

from src.api.database.research_db import ResearchDatabase
from src.api.routes import research
from src.audio.audio_processor import AudioProcessor
from src.settings import Settings
from src.utils.logging import websocket_logger
from src.voice_recognition.speech_to_text import SpeechToText
from src.voice_recognition.wake_word import WakeWordDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Create FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.mongodb_client = AsyncIOMotorClient("mongodb://localhost:27017")
    app.research_db = ResearchDatabase(app.mongodb_client)
    yield
    # Shutdown
    app.mongodb_client.close()


app = FastAPI(
    title="GoodRobot API",
    description="API for GoodRobot voice AI assistant",
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None,  # Disable default docs URL
    redoc_url=None  # Disable default redoc URL
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

    @field_validator("port")
    @classmethod
    def validate_port(cls, v):
        if v < 1024 or v > 65535:
            raise ValueError("Port number must be between 1024 and 65535")
        return v

    @field_validator("domain")
    @classmethod
    def validate_domain(cls, v):
        if not v or len(v) > 255:
            raise ValueError("Domain must be between 1 and 255 characters")
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


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the landing page."""
    return templates.TemplateResponse(
        "landing.html",
        {"request": request}
    )


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
async def process_voice(
    request: Request, audio: UploadFile = File(...), language: str = Form(...)
):
    try:
        contents = await audio.read()
        if not contents:
            raise CustomHTTPException(
                status_code=400,
                detail="Empty audio file",
                log_message="Received empty audio file",
            )

        # Process audio data
        audio_array = np.frombuffer(contents, dtype=np.int16)
        wake_word_detected = wake_word_detector.process_audio_chunk(contents)

        if not wake_word_detected:
            return {"wake_word_detected": False}

        transcription = await speech_to_text.transcribe_audio(contents)
        return {"wake_word_detected": True, "transcription": transcription}

    except CustomHTTPException as e:
        raise e
    except Exception as e:
        raise CustomHTTPException(
            status_code=500,
            detail="Internal server error",
            log_message=f"Error processing voice: {str(e)}",
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

            # Stop audio processing first
            if client_id in self.audio_processors:
                try:
                    await self.audio_processors[client_id].stop()
                    await self.audio_processors[client_id].cleanup()
                except Exception as e:
                    self.logger.logger.error(f"Error stopping audio processor for client {client_id}: {e}")
                finally:
                    del self.audio_processors[client_id]

            # Close WebSocket connection
            try:
                await self.active_connections[client_id].close()
            except Exception as e:
                self.logger.logger.error(f"Error closing WebSocket for client {client_id}: {e}")
            finally:
                del self.active_connections[client_id]

            self.logger.logger.info(f"Client {client_id} disconnected and resources cleaned up")

    async def send_json_message(self, client_id: str, message: dict):
        """Send a JSON message to a specific client"""
        try:
            if client_id in self.active_connections:
                await self.active_connections[client_id].send_json(message)
        except Exception as e:
            self.logger.logger.error(
                f"Error sending JSON message to client {client_id}: {e}"
            )

    async def process_json_message(self, message: dict, client_id: str):
        """Process incoming JSON message from client"""
        try:
            if client_id not in self.active_connections:
                logger.warning(f"Received message for unknown client: {client_id}")
                return

            # Message is already parsed, no need to load it again
            if not isinstance(message, dict):
                logger.error(f"Invalid message format: {type(message)}")
                return

            message_type = message.get("type")
            if not message_type:
                logger.error("Message missing type field")
                return

            if message_type == "config":
                # Handle configuration message
                config = message.get("config", {})
                if client_id in self.audio_processors:
                    # Ensure configure is awaited
                    await self.audio_processors[client_id].configure(config)
                    await self.send_json_message(
                        client_id, {"type": "config_ack", "status": "success"}
                    )
            elif message_type == "process_calibration":
                # Handle calibration samples
                samples = message.get("samples", [])
                if not samples:
                    await self._send_error(client_id, "No calibration samples provided")
                    return
                
                try:
                    # Add samples to wake word detector
                    for sample in samples:
                        wake_word_detector.add_calibration_sample(sample)
                    
                    # Send calibration complete message with new settings
                    await self.send_json_message(
                        client_id,
                        {
                            "type": "calibration_complete",
                            "sensitivity": wake_word_detector.sensitivity,
                            "noise_threshold": wake_word_detector.noise_threshold
                        }
                    )
                except Exception as e:
                    logger.error(f"Error processing calibration samples: {str(e)}")
                    await self._send_error(client_id, f"Calibration error: {str(e)}")
            elif message_type == "command":
                # Handle command message
                command = message.get("command")
                if command == "start":
                    if client_id in self.audio_processors:
                        await self.audio_processors[client_id].start()
                        await self.send_json_message(client_id, {"type": "command_ack", "command": "start", "status": "success"})
                elif command == "stop":
                    if client_id in self.audio_processors:
                        await self.audio_processors[client_id].stop()
                        await self.send_json_message(client_id, {"type": "command_ack", "command": "stop", "status": "success"})
                        # Initiate cleanup after stop
                        await self.disconnect(client_id)
            else:
                logger.warning(f"Unknown message type: {message_type}")

        except Exception as e:
            logger.error(f"Error processing JSON message: {str(e)}")
            logger.error(traceback.format_exc())
            await self._send_error(client_id, f"Error processing message: {str(e)}")

    async def process_binary_message(self, message: bytes, client_id: str):
        try:
            if client_id not in self.audio_processors:
                raise ValueError("No audio processor found for client")

            processor = self.audio_processors[client_id]

            # Process for wake word detection
            audio_array = np.frombuffer(message, dtype=np.int16)
            wake_word_detected = wake_word_detector.process_audio_chunk(audio_array)

            if wake_word_detected:
                self.logger.logger.info(f"Wake word detected for client {client_id}")
                await self._send_wake_word_detected(client_id)

            # Process audio data
            result = await processor.process_audio(message)

            # Send processing results
            await self.active_connections[client_id].send_json(
                {
                    "type": "audio_processed",
                    "data": result,
                    "wake_word_detected": wake_word_detected,
                }
            )

        except Exception as e:
            self.logger.logger.error(
                f"Error processing binary message from client {client_id}: {e}"
            )
            await self._send_error(client_id, "Error processing audio data")

    async def _handle_config_message(self, data: dict, client_id: str):
        try:
            if client_id in self.audio_processors:
                await self.audio_processors[client_id].configure(data.get("config", {}))
                self.logger.logger.info(f"Updated configuration for client {client_id}")
        except Exception as e:
            self.logger.logger.error(
                f"Error configuring audio processor for client {client_id}: {e}"
            )
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
                self.logger.logger.warning(
                    f"Unknown command from client {client_id}: {command}"
                )
        except Exception as e:
            self.logger.logger.error(
                f"Error handling command for client {client_id}: {e}"
            )
            await self._send_error(client_id, "Command processing error")

    async def _handle_memory_message(self, data: dict, client_id: str):
        try:
            memory = Memory(
                text=data.get("text"),
                timestamp=data.get("timestamp"),
                tags=data.get("tags", ["user_input", "direct_memory"]),
            )
            memories.append(memory.dict())
            # Keep only the last 50 memories
            if len(memories) > 50:
                memories.pop(0)

            self.logger.logger.info(
                f"Received memory update from client {client_id}: {memory.text}"
            )

            # Send updated memories back to client
            await self.active_connections[client_id].send_json(
                {"type": "memory_update", "memories": memories}
            )
        except Exception as e:
            self.logger.logger.error(
                f"Error handling memory message from client {client_id}: {e}"
            )
            await self._send_error(client_id, "Memory processing error")

    async def _send_wake_word_detected(self, client_id: str):
        try:
            if client_id in self.active_connections:
                await self.active_connections[client_id].send_json(
                    {"type": "wake_word", "wake_word_detected": True}
                )
        except Exception as e:
            self.logger.logger.error(
                f"Error sending wake word detection to client {client_id}: {e}"
            )

    async def _send_error(self, client_id: str, message: str):
        try:
            if client_id in self.active_connections:
                await self.active_connections[client_id].send_json(
                    {"type": "error", "message": message}
                )
        except Exception as e:
            self.logger.logger.error(
                f"Error sending error message to client {client_id}: {e}"
            )


# Initialize WebSocket manager
websocket_manager = WebSocketManager()


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    ws_handler = await websocket_manager.connect(websocket, client_id)

    try:
        while True:
            try:
                # Try to receive JSON message first
                data = await websocket.receive_json()
                # Process the JSON message directly without trying to parse it again
                await websocket_manager.process_json_message(data, client_id)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to receive binary data
                try:
                    data = await websocket.receive_bytes()
                    await websocket_manager.process_binary_message(data, client_id)
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    logger.error(f"Error receiving data: {str(e)}")
                    logger.error(traceback.format_exc())
                    break
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for client {client_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket connection for client {client_id}: {e}")
        logger.error(traceback.format_exc())
    finally:
        await websocket_manager.disconnect(client_id, ws_handler)


@app.websocket("/ws")
async def websocket_endpoint_general(websocket: WebSocket):
    client_id = f"client_{uuid.uuid4().hex[:8]}"
    await websocket_endpoint(websocket, client_id)


@app.websocket("/voice")
async def websocket_endpoint_voice(websocket: WebSocket):
    client_id = f"voice_{uuid.uuid4().hex[:8]}"
    await websocket_endpoint(websocket, client_id)


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
    return templates.TemplateResponse("research.html", {"request": request})


@app.get("/sample-recorder")
async def sample_recorder(request: Request):
    """Serve the sample recorder UI"""
    return templates.TemplateResponse("sample_recorder.html", {"request": request})


@app.post("/api/process_sample")
async def process_sample(request: Request):
    """Process an audio sample and save it if it contains the wake word"""
    try:
        form = await request.form()
        audio_file = form.get("audio")
        sample_type = form.get("type", "human")
        
        if not audio_file:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "No audio file provided"}
            )
        
        # Read audio data
        audio_data = await audio_file.read()
        
        try:
            # First try reading with soundfile
            audio_array, sample_rate = sf.read(io.BytesIO(audio_data))
            logger.info(f"Successfully read audio with soundfile: {sample_rate}Hz")
        except Exception as e:
            logger.warning(f"Failed to read with soundfile: {str(e)}, trying ffmpeg conversion")
            # If soundfile fails, use ffmpeg to convert from webm to wav
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as webm_file, \
                 tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as wav_file:
                
                # Write webm data to temp file
                webm_file.write(audio_data)
                webm_file.flush()
                
                # Convert webm to wav using ffmpeg
                try:
                    subprocess.run([
                        'ffmpeg',
                        '-i', webm_file.name,
                        '-acodec', 'pcm_s16le',
                        '-ar', '16000',
                        '-ac', '1',
                        '-f', 'wav',
                        wav_file.name
                    ], check=True, capture_output=True)
                    
                    # Read the converted wav file
                    audio_array, sample_rate = sf.read(wav_file.name)
                    logger.info(f"Successfully converted and read audio: {sample_rate}Hz")
                except subprocess.CalledProcessError as e:
                    logger.error(f"FFmpeg conversion failed: {e.stderr.decode()}")
                    raise
                finally:
                    # Clean up temp files
                    try:
                        os.unlink(webm_file.name)
                        os.unlink(wav_file.name)
                    except Exception as e:
                        logger.warning(f"Error cleaning up temp files: {str(e)}")
        
        # Convert to mono if stereo
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            audio_array = signal.resample(audio_array, int(len(audio_array) * 16000 / sample_rate))
            sample_rate = 16000
            logger.info(f"Resampled audio to 16kHz")
        
        # Normalize audio to [-1, 1] range
        audio_array = audio_array / np.max(np.abs(audio_array))
        
        # Convert to int16 for wake word detector
        audio_array_int16 = (audio_array * 32767).astype(np.int16)
        
        # Calculate audio quality metrics
        metrics = calculate_audio_metrics(audio_array, sample_rate)
        
        # Process with wake word detector
        detected = wake_word_detector.process_audio_chunk(audio_array_int16.tobytes())
        metrics['wake_word'] = detected
        
        # Validate sample
        validation_result = validate_sample(audio_array, sample_rate, metrics)
        
        if validation_result['is_valid']:
            # Save the sample
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            directory = os.path.join("tests", "audio_samples", "wake_word", sample_type)
            os.makedirs(directory, exist_ok=True)
            
            filename = os.path.join(directory, f"jarvis_{sample_type}_{timestamp}.wav")
            sf.write(filename, audio_array_int16, sample_rate)
            
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "Sample saved successfully",
                    "metrics": metrics
                }
            )
        else:
            return JSONResponse(
                status_code=200,
                content={
                    "success": False,
                    "error": validation_result['error'],
                    "metrics": metrics
                }
            )
            
    except Exception as e:
        logger.error(f"Error processing sample: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

def calculate_audio_metrics(audio_array: np.ndarray, sample_rate: int) -> Dict:
    """Calculate various audio quality metrics."""
    try:
        # Ensure audio is in float32 format and normalized to [-1, 1]
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)
            if audio_array.dtype == np.int16:
                audio_array = audio_array / 32767.0
        
        # Apply dynamic range compression to boost low-level signals
        threshold = 0.1  # Threshold for compression
        ratio = 2.0  # Compression ratio
        compressed_audio = np.copy(audio_array)
        mask = np.abs(audio_array) < threshold
        compressed_audio[mask] = np.sign(audio_array[mask]) * (np.abs(audio_array[mask]) ** (1/ratio))
        
        # Calculate RMS volume level with increased sensitivity
        rms = np.sqrt(np.mean(np.square(compressed_audio)))
        volume_level = min(100, int(rms * 400))  # Increased multiplier for better sensitivity
        
        # Calculate signal-to-noise ratio (SNR) using a more robust method
        # Split audio into frames for better noise estimation
        frame_length = int(sample_rate * 0.02)  # 20ms frames
        frames = np.array_split(compressed_audio, len(compressed_audio) // frame_length)
        
        # Calculate energy of each frame
        frame_energies = [np.mean(np.square(frame)) for frame in frames]
        
        # Find the quietest 20% of frames as noise reference
        noise_frames = int(len(frames) * 0.2)
        noise_energy = np.mean(sorted(frame_energies)[:noise_frames])
        
        # Calculate signal energy from the loudest 20% of frames
        signal_frames = int(len(frames) * 0.2)
        signal_energy = np.mean(sorted(frame_energies)[-signal_frames:])
        
        # Calculate SNR with a minimum noise floor
        noise_floor = 1e-6  # Minimum noise floor to prevent division by zero
        snr = 10 * np.log10(signal_energy / max(noise_energy, noise_floor))
        
        # Calculate clarity score based on zero-crossings and spectral features
        zero_crossings = np.sum(np.diff(np.signbit(compressed_audio).astype(int)))
        zcr_score = min(100, int((zero_crossings / len(compressed_audio)) * sample_rate * 0.1))
        
        # Calculate spectral centroid for clarity
        fft = np.fft.rfft(compressed_audio)
        magnitude = np.abs(fft)
        frequency = np.fft.rfftfreq(len(compressed_audio), 1/sample_rate)
        spectral_centroid = np.sum(frequency * magnitude) / np.sum(magnitude)
        spectral_score = min(100, int(spectral_centroid / 1000))  # Normalize to 0-100
        
        # Combine scores for final clarity
        clarity_score = int((zcr_score + spectral_score) / 2)
        
        # Detect potential clipping
        clipping_threshold = 0.95  # 95% of max amplitude
        clipping_samples = np.sum(np.abs(compressed_audio) > clipping_threshold)
        clipping_percentage = (clipping_samples / len(compressed_audio)) * 100
        
        return {
            "volume_level": volume_level,  # 0-100
            "signal_to_noise": min(100, max(0, int(snr))),  # 0-100
            "clarity": clarity_score,  # 0-100
            "clipping_percentage": round(clipping_percentage, 2),  # percentage of clipped samples
            "quality_score": min(100, int((volume_level + min(100, max(0, int(snr))) + clarity_score) / 3))
        }
    except Exception as e:
        logger.error(f"Error calculating audio metrics: {str(e)}")
        return {
            "volume_level": 0,
            "signal_to_noise": 0,
            "clarity": 0,
            "clipping_percentage": 0,
            "quality_score": 0
        }

def validate_sample(audio_array: np.ndarray, sample_rate: int, metrics: dict) -> dict:
    """Validate the audio sample against quality criteria"""
    result = {
        'is_valid': True,
        'error': None
    }
    
    # Check duration
    duration = len(audio_array) / sample_rate
    if duration < 0.5 or duration > 15.0:
        result['is_valid'] = False
        result['error'] = f"Invalid duration: {duration:.2f}s (must be between 0.5s and 15.0s)"
        return result
    
    # Check volume level - lowered from 1 to 0.5
    if metrics['volume_level'] < 0.5:
        result['is_valid'] = False
        result['error'] = f"Volume too low: {metrics['volume_level']}% (minimum 0.5%)"
        return result
    
    # Check noise level - increased from 90 to 95 and added minimum threshold
    if metrics['signal_to_noise'] > 95 or metrics['signal_to_noise'] < 5:
        result['is_valid'] = False
        result['error'] = f"Invalid noise level: SNR {metrics['signal_to_noise']}dB (must be between 5dB and 95dB)"
        return result
    
    # Check clarity - lowered from 1 to 0.5
    if metrics['clarity'] < 0.5:
        result['is_valid'] = False
        result['error'] = f"Audio not clear enough: clarity score {metrics['clarity']}%"
        return result
    
    # Check wake word detection
    if not metrics['wake_word']:
        result['is_valid'] = False
        result['error'] = "Wake word not detected"
        return result
    
    return result

@app.post("/api/save_sample")
async def save_sample(
    audio_data: bytes = File(...),
    sample_type: str = Form(...),
    timestamp: str = Form(...)
):
    try:
        # Create samples directory if it doesn't exist
        samples_dir = Path("samples")
        samples_dir.mkdir(exist_ok=True)
        
        # Generate unique filename
        filename = f"{sample_type}_{timestamp}.wav"
        file_path = samples_dir / filename
        
        # Convert audio data to numpy array for processing
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        sample_rate = 16000  # Standard sample rate
        
        # Calculate audio metrics
        metrics = calculate_audio_metrics(audio_array, sample_rate)
        
        # Generate waveform data (downsampled for visualization)
        waveform = generate_waveform_data(audio_array)
        
        # Calculate duration
        duration = len(audio_array) / sample_rate
        
        # Save the audio file
        with wave.open(str(file_path), 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_array.tobytes())
        
        # Save metadata alongside the audio file
        metadata = {
            "filename": filename,
            "type": sample_type,
            "timestamp": timestamp,
            "duration": duration,
            "metrics": metrics,
            "waveform": waveform.tolist()
        }
        
        metadata_path = file_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        return {"status": "success", "filename": filename, "metadata": metadata}
    except Exception as e:
        logger.error(f"Error saving sample: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def generate_waveform_data(audio_array: np.ndarray, num_points: int = 100) -> np.ndarray:
    """Generate downsampled waveform data for visualization."""
    if len(audio_array) == 0:
        return np.zeros(num_points)
    
    # Normalize audio to [-1, 1]
    audio_normalized = audio_array / 32767.0
    
    # Calculate points per segment
    points_per_segment = len(audio_array) // num_points
    
    if points_per_segment < 1:
        return np.zeros(num_points)
    
    # Calculate min and max for each segment
    waveform = []
    for i in range(num_points):
        start = i * points_per_segment
        end = start + points_per_segment
        segment = audio_normalized[start:end]
        if len(segment) > 0:
            waveform.append(np.mean(segment))
        else:
            waveform.append(0)
    
    return np.array(waveform)

@app.get("/api/list_samples")
async def list_samples():
    try:
        samples_dir = Path("samples")
        if not samples_dir.exists():
            return {"samples": []}
        
        samples = []
        for wav_file in samples_dir.glob("*.wav"):
            metadata_file = wav_file.with_suffix('.json')
            
            if metadata_file.exists():
                # Load metadata if it exists
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                samples.append(metadata)
            else:
                # Generate basic metadata if no metadata file exists
                name_parts = wav_file.stem.split("_")
                sample_type = name_parts[0]
                timestamp = "_".join(name_parts[1:])
                
                # Read audio file for basic metrics
                with wave.open(str(wav_file), 'rb') as wav:
                    frames = wav.readframes(wav.getnframes())
                    audio_array = np.frombuffer(frames, dtype=np.int16)
                    duration = wav.getnframes() / wav.getframerate()
                
                metrics = calculate_audio_metrics(audio_array, wav.getframerate())
                waveform = generate_waveform_data(audio_array)
                
                metadata = {
                    "filename": wav_file.name,
                    "type": sample_type,
                    "timestamp": timestamp,
                    "duration": duration,
                    "metrics": metrics,
                    "waveform": waveform.tolist()
                }
                
                # Save metadata for future use
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f)
                
                samples.append(metadata)
        
        return {"samples": samples}
    except Exception as e:
        logger.error(f"Error listing samples: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/delete_samples")
async def delete_samples(filenames: List[str]):
    """Delete multiple samples at once."""
    try:
        samples_dir = Path("samples")
        deleted = []
        errors = []
        
        for filename in filenames:
            try:
                wav_file = samples_dir / filename
                metadata_file = wav_file.with_suffix('.json')
                
                if wav_file.exists():
                    wav_file.unlink()
                if metadata_file.exists():
                    metadata_file.unlink()
                    
                deleted.append(filename)
            except Exception as e:
                errors.append({"filename": filename, "error": str(e)})
        
        return {
            "status": "success",
            "deleted": deleted,
            "errors": errors
        }
    except Exception as e:
        logger.error(f"Error deleting samples: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/get_sample/{filename}")
async def get_sample(filename: str):
    """Get a specific audio sample file."""
    try:
        # Ensure the filename is safe
        safe_filename = os.path.basename(filename)
        file_path = os.path.join("tests", "audio_samples", "wake_word", "human", safe_filename)
        
        if not os.path.exists(file_path):
            # Try AI samples directory
            file_path = os.path.join("tests", "audio_samples", "wake_word", "ai", safe_filename)
            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail="Sample not found")
        
        return FileResponse(
            file_path,
            media_type="audio/wav",
            filename=safe_filename
        )
    except Exception as e:
        logger.error(f"Error getting sample {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/docs", response_class=HTMLResponse)
async def custom_docs(request: Request):
    """Serve the custom API documentation page."""
    return templates.TemplateResponse("docs.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.api:app", host="0.0.0.0", port=8000, reload=True)
