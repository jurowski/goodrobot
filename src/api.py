from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import os
from dotenv import load_dotenv
from src.settings import Settings
from src.voice_recognition.speech_to_text import SpeechToText
from src.voice_recognition.wake_word import WakeWordDetector
import json
import logging
from fastapi import WebSocketDisconnect
import asyncio
from typing import Dict, Set
import numpy as np
import io
import soundfile as sf
import time
import traceback
import wave

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="Voice AI Assistant")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directory if it doesn't exist
os.makedirs("static", exist_ok=True)
os.makedirs("static/utilities", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

# Add route to serve audio-processor.js directly
@app.get("/audio-processor.js")
async def serve_audio_processor():
    """Serve the audio processor worklet with proper headers."""
    response = FileResponse(
        "static/audio-processor.js",
        media_type="application/javascript",
        headers={
            "Cache-Control": "no-cache",
            "Cross-Origin-Opener-Policy": "same-origin",
            "Cross-Origin-Embedder-Policy": "require-corp"
        }
    )
    return response

# Add route to serve audio-processor.js from static directory
@app.get("/static/audio-processor.js")
async def serve_static_audio_processor():
    """Serve the audio processor worklet from static directory."""
    return await serve_audio_processor()

# Add route to serve the utilities page
@app.get("/utilities/speech-to-text")
async def serve_speech_to_text():
    """Serve the speech to text testing tool."""
    file_path = os.path.join(os.getcwd(), "static/utilities/speech-to-text.html")
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    
    logger.info(f"Serving file: {file_path}")
    return FileResponse(
        file_path,
        media_type="text/html",
        headers={
            "Cache-Control": "no-cache"
        }
    )

# Initialize settings and services
settings = Settings()
stt = SpeechToText(settings)
wake_word_detector = WakeWordDetector(settings)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Log working directory
logger.info(f"Working directory: {os.getcwd()}")
logger.info(f"Static directory exists: {os.path.exists('static')}")
logger.info(f"Utilities directory exists: {os.path.exists('static/utilities')}")
logger.info(f"Speech-to-text file exists: {os.path.exists('static/utilities/speech-to-text.html')}")

# Store active connections
active_connections: Set[WebSocket] = set()

@app.get("/")
async def get_index():
    """Serve the homepage."""
    return FileResponse("static/index.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.add(websocket)
    logger.info("New WebSocket connection established")
    
    # Buffer to store audio chunks
    audio_buffer = io.BytesIO()
    wav_file = wave.open(audio_buffer, 'wb')
    wav_file.setnchannels(1)  # Mono
    wav_file.setsampwidth(2)  # 16-bit
    wav_file.setframerate(16000)  # 16kHz
    
    try:
        while True:
            # Receive audio data
            data = await websocket.receive_bytes()
            
            # Process for wake word detection
            try:
                # The data is already in int16 format from the AudioWorklet
                audio_array = np.frombuffer(data, dtype=np.int16)
                
                # No need to truncate or pad, we're receiving exactly 512 samples
                wake_word_detected = wake_word_detector.process_audio_chunk(audio_array)
                if wake_word_detected:
                    logger.info("Wake word detected!")
                    await websocket.send_json({
                        "type": "wake_word",
                        "wake_word_detected": True
                    })
            except Exception as e:
                logger.error(f"Error in wake word detection: {str(e)}")
                logger.error(traceback.format_exc())
            
            # Write to WAV buffer for transcription
            wav_file.writeframes(data)
            
            # If buffer size exceeds threshold, transcribe
            if audio_buffer.tell() > 32000:  # Adjust threshold as needed
                try:
                    # Close the current WAV file
                    wav_file.close()
                    
                    # Get the buffer content as bytes
                    audio_buffer.seek(0)
                    audio_bytes = audio_buffer.getvalue()
                    
                    # Transcribe audio
                    text = await stt.transcribe_audio(audio_bytes)
                    
                    # Send transcription back to client
                    await websocket.send_json({
                        "type": "transcription",
                        "text": text
                    })
                except Exception as e:
                    logger.error(f"Error in transcription: {str(e)}")
                    logger.error(traceback.format_exc())
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })
                finally:
                    # Clear buffer and start new WAV file
                    audio_buffer = io.BytesIO()
                    wav_file = wave.open(audio_buffer, 'wb')
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(16000)
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {e}")
        logger.error(traceback.format_exc())
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)
        try:
            await websocket.close()
        except:
            pass
        logger.info("WebSocket connection closed")
        wav_file.close()

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe audio file to text.
    Accepts WAV files.
    """
    try:
        # Read the audio file
        contents = await file.read()
        
        # Transcribe the audio
        transcription = stt.transcribe_audio(contents)
        
        return {"text": transcription}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    """Check if the API is running."""
    return {"status": "healthy"}

# Cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down API...")
    # Close all active WebSocket connections
    for connection in active_connections:
        try:
            await connection.close()
        except Exception as e:
            logger.error(f"Error closing WebSocket connection: {str(e)}")
    
    # Cleanup resources
    wake_word_detector.cleanup()
    stt.cleanup()
    logger.info("API shutdown complete") 