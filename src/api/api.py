"""
Main API module for GoodRobot.
This module sets up the FastAPI application and includes all API routes.
"""

from fastapi import FastAPI, UploadFile, File, Form, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import json
import logging
import os
from pathlib import Path
import asyncio
import numpy as np
import soundfile as sf
import io

from src.voice_recognition.wake_word import WakeWordDetector
from src.voice_recognition.speech_to_text import SpeechToText
from src.settings import Settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="GoodRobot API",
    description="API for GoodRobot voice AI assistant",
    version="0.1.0"
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
app.mount("/static", StaticFiles(directory=str(project_root / "src" / "api" / "static")), name="static")

# Initialize settings
settings = Settings()

# Initialize wake word detector and speech-to-text
wake_word_detector = WakeWordDetector(settings)
speech_to_text = SpeechToText(settings)

@app.get("/")
async def root():
    return {"message": "Welcome to GoodRobot API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/voice", response_class=HTMLResponse)
async def voice_interface(request: Request):
    """Serve the voice interface HTML page."""
    return templates.TemplateResponse(
        "voice.html",
        {"request": request}
    )

@app.post("/voice")
async def process_voice(
    audio: UploadFile = File(...),
    language: str = Form(...)
):
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
            
            return {
                "transcription": transcription,
                "response": response
            }
        else:
            return {
                "transcription": "No wake word detected",
                "response": "Please say the wake word first"
            }
    except Exception as e:
        logger.error(f"Error processing voice: {str(e)}")
        return {"error": str(e)}

@app.websocket("/ws/voice")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time voice processing."""
    await websocket.accept()
    
    try:
        while True:
            # Receive audio data
            audio_data = await websocket.receive_bytes()
            
            # Send debug log
            await websocket.send_text(json.dumps({
                "type": "log",
                "data": f"Processing audio chunk of size {len(audio_data)} bytes"
            }))
            
            # Check for wake word
            try:
                wake_word_detected = wake_word_detector.process_audio_chunk(audio_data)
                
                # Send debug log
                await websocket.send_text(json.dumps({
                    "type": "log",
                    "data": f"Wake word detection result: {wake_word_detected}"
                }))
                
                if wake_word_detected:
                    # Send wake word detected message
                    await websocket.send_text(json.dumps({
                        "type": "wake_word",
                        "data": "Wake word detected"
                    }))
                    
                    # Send debug log
                    await websocket.send_text(json.dumps({
                        "type": "log",
                        "data": "Starting transcription..."
                    }))
                    
                    # Transcribe audio
                    transcription = await speech_to_text.transcribe_audio(audio_data)
                    
                    # Send transcription back
                    await websocket.send_text(json.dumps({
                        "type": "transcription",
                        "data": transcription
                    }))
                    
                    # Send debug log
                    await websocket.send_text(json.dumps({
                        "type": "log",
                        "data": f"Transcription completed: {transcription}"
                    }))
                    
                    # TODO: Process the transcription and generate response
                    response = f"Transcription: {transcription}"
                    
                    # Send response back
                    await websocket.send_text(json.dumps({
                        "type": "response",
                        "data": response
                    }))
                else:
                    # Send wake word not detected message
                    await websocket.send_text(json.dumps({
                        "type": "wake_word",
                        "data": "No wake word detected"
                    }))
            except Exception as e:
                # Send error log
                await websocket.send_text(json.dumps({
                    "type": "log",
                    "data": f"Error processing audio: {str(e)}"
                }))
                logger.error(f"Error processing audio: {str(e)}")
                
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        await websocket.close()

# Ensure required directories exist
def ensure_directories():
    """Create required directories if they don't exist."""
    directories = [
        project_root / "src" / "api" / "templates",
        project_root / "src" / "api" / "static",
        project_root / "src" / "api" / "static" / "images"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory exists: {directory.exists()}")

# Initialize directories
ensure_directories() 