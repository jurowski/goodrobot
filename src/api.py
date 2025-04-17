from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
import os
from dotenv import load_dotenv
from src.settings import Settings
from src.voice_recognition.speech_to_text import SpeechToText
from src.voice_recognition.wake_word import WakeWordDetector
import json
import logging
from fastapi import WebSocketDisconnect
import asyncio
from typing import Dict, Set, Optional, List
import numpy as np
import io
import soundfile as sf
import time
import traceback
import wave
import requests
from pydantic import BaseModel
from datetime import datetime

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

# Store active connections and memories
active_connections: Set[WebSocket] = set()
memories: List[Dict] = []

class Memory(BaseModel):
    text: str
    timestamp: str
    tags: List[str]

class TextToSpeechRequest(BaseModel):
    text: str
    voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # Default voice ID
    api_key: str

@app.get("/")
async def get_index():
    """Serve the homepage."""
    return FileResponse("static/index.html")

def generate_response(question: str) -> str:
    """Generate a response based on the question."""
    question = question.lower()
    
    # Handle time-related questions
    if any(time_word in question for time_word in ['time', 'what time', 'current time', 'clock']):
        current_time = datetime.now().strftime("%I:%M %p")
        return f"The current time is {current_time}."
    
    # Handle date-related questions
    if any(date_word in question for date_word in ['date', 'what date', 'today\'s date', 'what day is it']):
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        return f"Today is {current_date}."
    
    # Handle greeting questions
    if any(greeting in question for greeting in ['hello', 'hi', 'hey', 'how are you']):
        return "Hello! I'm doing well, thank you for asking. How can I help you today?"
    
    # Handle identity questions
    if any(identity in question for identity in ['who are you', 'what are you', 'your name']):
        return "I am Jarvis, your AI assistant. I'm here to help you with tasks, answer questions, and provide assistance."
    
    # Handle capability questions
    if any(capability in question for capability in ['what can you do', 'help', 'capabilities']):
        return "I can help you with various tasks including telling time, answering questions, and providing information. I'm still learning new capabilities!"
    
    # Default response for other questions
    return "I'm not sure how to help with that yet. Could you please rephrase your question?"

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
            try:
                # Try to receive JSON message first
                data = await websocket.receive_json()
                
                # Handle JSON messages
                if data.get('type') == 'memory':
                    # Handle memory update
                    memory = Memory(
                        text=data.get('text'),
                        timestamp=data.get('timestamp'),
                        tags=data.get('tags', ['user_input', 'direct_memory'])
                    )
                    memories.append(memory.dict())
                    # Keep only the last 50 memories
                    if len(memories) > 50:
                        memories.pop(0)
                    
                    logger.info(f"Received memory update: {memory.text}")
                    # Send updated memories back to client
                    await websocket.send_json({
                        "type": "memory_update",
                        "memories": memories
                    })
                elif data.get('type') == 'question':
                    # Handle question
                    question = data.get('text')
                    logger.info(f"Received question: {question}")
                    
                    # Add question to memories
                    memory = Memory(
                        text=question,
                        timestamp=data.get('timestamp'),
                        tags=['question', 'user_input']
                    )
                    memories.append(memory.dict())
                    # Keep only the last 50 memories
                    if len(memories) > 50:
                        memories.pop(0)
                    
                    # Generate response
                    response_text = generate_response(question)
                    
                    # Add response to memories
                    response_memory = Memory(
                        text=response_text,
                        timestamp=datetime.now().isoformat(),
                        tags=['response', 'assistant']
                    )
                    memories.append(response_memory.dict())
                    if len(memories) > 50:
                        memories.pop(0)
                    
                    # Log the response
                    logger.info(f"Generated response: {response_text}")
                    
                    # Send response
                    await websocket.send_json({
                        "type": "response",
                        "text": response_text
                    })
                    
                    # Send updated memories
                    await websocket.send_json({
                        "type": "memory_update",
                        "memories": memories
                    })
                elif data.get('type') == 'transcription':
                    # Handle simulated transcription
                    logger.info(f"Received simulated transcription: {data.get('text')}")
                    # Add transcription to memories
                    memory = Memory(
                        text=data.get('text'),
                        timestamp=data.get('timestamp'),
                        tags=['transcription', 'voice_input']
                    )
                    memories.append(memory.dict())
                    # Keep only the last 50 memories
                    if len(memories) > 50:
                        memories.pop(0)
                    
                    # Process transcription here
                    await websocket.send_json({
                        "type": "transcription",
                        "text": data.get('text')
                    })
                    # Send updated memories
                    await websocket.send_json({
                        "type": "memory_update",
                        "memories": memories
                    })
                
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected")
                break
            except json.JSONDecodeError:
                # If JSON parsing fails, try to receive binary data
                try:
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
                            
                            # Add transcription to memories
                            memory = Memory(
                                text=text,
                                timestamp=datetime.now().isoformat(),
                                tags=['transcription', 'voice_input']
                            )
                            memories.append(memory.dict())
                            # Keep only the last 50 memories
                            if len(memories) > 50:
                                memories.pop(0)
                            
                            # Send transcription back to client
                            await websocket.send_json({
                                "type": "transcription",
                                "text": text
                            })
                            # Send updated memories
                            await websocket.send_json({
                                "type": "memory_update",
                                "memories": memories
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
                    break
                except Exception as e:
                    logger.error(f"Error receiving data: {str(e)}")
                    logger.error(traceback.format_exc())
                    break
                
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

@app.post("/speak")
async def speak_text(request: TextToSpeechRequest):
    try:
        # Validate API key
        if not request.api_key:
            raise HTTPException(status_code=400, detail="API key is required")

        # Prepare request to ElevenLabs API
        headers = {
            "xi-api-key": request.api_key,
            "Content-Type": "application/json"
        }
        
        data = {
            "text": request.text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }

        # Make request to ElevenLabs API
        response = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{request.voice_id}",
            headers=headers,
            json=data
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"ElevenLabs API error: {response.text}"
            )

        # Return the audio data as a streaming response
        return StreamingResponse(
            io.BytesIO(response.content),
            media_type="audio/mpeg"
        )

    except Exception as e:
        logger.error(f"Error in text-to-speech: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/voices")
async def get_voices(api_key: str):
    try:
        # Validate API key
        if not api_key:
            raise HTTPException(status_code=400, detail="API key is required")

        # Get available voices from ElevenLabs
        headers = {"xi-api-key": api_key}
        response = requests.get(
            "https://api.elevenlabs.io/v1/voices",
            headers=headers
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"ElevenLabs API error: {response.text}"
            )

        return response.json()

    except Exception as e:
        logger.error(f"Error getting voices: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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