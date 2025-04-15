from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
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

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

# Initialize settings and services
settings = Settings()
stt = SpeechToText(settings)
wake_word_detector = WakeWordDetector(settings)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Store active connections
active_connections: Set[WebSocket] = set()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.add(websocket)
    logger.info("New WebSocket connection established")
    
    try:
        while True:
            try:
                # Receive audio data
                audio_data = await websocket.receive_bytes()
                logger.debug(f"Received audio chunk of size: {len(audio_data)} bytes")
                
                # Convert to numpy array (already in correct format from frontend)
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                
                # Process with wake word detector
                wake_word_detected = wake_word_detector.process_audio_chunk(audio_array)
                
                if wake_word_detected:
                    logger.info("Wake word detected!")
                    await websocket.send_json({"wake_word_detected": True})
                    
                    # Start transcription after wake word
                    try:
                        transcription = await stt.transcribe_audio(audio_array)
                        if transcription:
                            await websocket.send_json({
                                "type": "transcription",
                                "text": transcription
                            })
                    except Exception as e:
                        logger.error(f"Transcription error: {str(e)}")
                        await websocket.send_json({
                            "type": "error",
                            "message": "Transcription failed"
                        })
                
            except Exception as e:
                logger.error(f"Error processing audio: {str(e)}")
                await websocket.send_json({
                    "type": "error",
                    "message": "Error processing audio"
                })
    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)
        try:
            await websocket.close()
        except:
            pass
        logger.info("WebSocket connection closed")

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the homepage."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Voice AI Assistant</title>
        <style>
            :root {
                --bg-primary: #1a1b1e;
                --bg-secondary: #2c2e33;
                --text-primary: #e4e6eb;
                --text-secondary: #b0b3b8;
                --accent: #4f46e5;
                --accent-hover: #6366f1;
                --shadow: rgba(0, 0, 0, 0.2);
                --success: #22c55e;
                --error: #ef4444;
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: var(--bg-primary);
                color: var(--text-primary);
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }

            .logo-container {
                text-align: center;
                margin-bottom: 40px;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                gap: 20px;
            }

            .logo {
                width: 200px;
                height: auto;
                filter: drop-shadow(0 0 10px rgba(79, 70, 229, 0.3));
                transition: filter 0.3s ease;
            }

            .logo:hover {
                filter: drop-shadow(0 0 15px rgba(79, 70, 229, 0.5));
            }
            
            h1 {
                color: var(--text-primary);
                text-align: center;
                margin: 0;
                font-size: 2.5rem;
                font-weight: 700;
                letter-spacing: -0.5px;
            }
            
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                padding: 20px;
            }
            
            .panel {
                background-color: var(--bg-secondary);
                border-radius: 12px;
                padding: 25px;
                box-shadow: 0 4px 6px var(--shadow);
                transition: all 0.3s ease;
                text-decoration: none;
                color: inherit;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            .panel:hover {
                transform: translateY(-5px);
                box-shadow: 0 8px 12px var(--shadow);
                border-color: var(--accent);
            }
            
            .panel h2 {
                color: var(--text-primary);
                margin-top: 0;
                margin-bottom: 15px;
                font-size: 1.5rem;
                font-weight: 600;
            }
            
            .panel p {
                color: var(--text-secondary);
                margin: 0 0 20px 0;
                line-height: 1.5;
                font-size: 1rem;
            }
            
            .api-link {
                display: inline-block;
                margin-top: 10px;
                color: var(--accent);
                text-decoration: none;
                font-size: 0.9rem;
                font-weight: 500;
                padding: 8px 12px;
                border-radius: 6px;
                background-color: rgba(79, 70, 229, 0.1);
                transition: all 0.2s ease;
            }
            
            .api-link:hover {
                background-color: rgba(79, 70, 229, 0.2);
                color: var(--accent-hover);
            }

            @media (max-width: 768px) {
                body {
                    padding: 10px;
                }
                
                h1 {
                    font-size: 2rem;
                }
                
                .grid {
                    padding: 10px;
                }
                
                .panel {
                    padding: 20px;
                }
            }

            .live-panel {
                grid-column: 1 / -1;
                text-align: center;
            }
            
            .status-indicator {
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                margin-right: 8px;
                background-color: var(--text-secondary);
            }
            
            .status-indicator.active {
                background-color: var(--success);
                animation: pulse 2s infinite;
            }
            
            .status-indicator.detected {
                background-color: var(--accent);
            }
            
            .status-indicator.error {
                background-color: var(--error);
            }
            
            .control-button {
                background-color: var(--accent);
                color: var(--text-primary);
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 1rem;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.2s ease;
                margin: 10px 0;
            }
            
            .control-button:hover {
                background-color: var(--accent-hover);
            }
            
            .control-button:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
            
            .transcript {
                margin-top: 20px;
                padding: 15px;
                background-color: var(--bg-secondary);
                border-radius: 8px;
                text-align: left;
                min-height: 100px;
                max-height: 200px;
                overflow-y: auto;
            }
            
            @keyframes pulse {
                0% { opacity: 0.8; }
                50% { opacity: 1; }
                100% { opacity: 0.8; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="logo-container">
                <img src="/assets/images/logo-good-robot-dark-1.png" alt="Good Robot Logo" class="logo">
                <h1>Voice AI Assistant</h1>
            </div>
            <div class="grid">
                <div class="panel live-panel">
                    <h2>Live Wake Word Detection</h2>
                    <p>Click start and say "Jarvis" to activate voice recognition.</p>
                    <div>
                        <span class="status-indicator" id="statusIndicator"></span>
                        <span id="statusText">Microphone inactive</span>
                    </div>
                    <button class="control-button" id="startButton">Start Listening</button>
                    <div class="transcript" id="transcript"></div>
                    <div id="debugLog" style="margin-top: 20px; font-family: monospace; font-size: 12px; color: var(--text-secondary);"></div>
                </div>
                <a href="/transcribe" class="panel">
                    <h2>Speech to Text</h2>
                    <p>Upload an audio file and get its transcription. Supports multiple audio formats and provides accurate text output.</p>
                    <span class="api-link">POST /transcribe</span>
                </a>
                <a href="/docs" class="panel">
                    <h2>API Documentation</h2>
                    <p>Interactive API documentation with Swagger UI. Test endpoints and explore available features.</p>
                    <span class="api-link">Swagger UI</span>
                </a>
                <a href="/redoc" class="panel">
                    <h2>API Reference</h2>
                    <p>Detailed API reference documentation with ReDoc. View comprehensive API specifications.</p>
                    <span class="api-link">ReDoc</span>
                </a>
                <a href="/health" class="panel">
                    <h2>Health Check</h2>
                    <p>Monitor the API's operational status. Verify if all services are functioning correctly.</p>
                    <span class="api-link">GET /health</span>
                </a>
            </div>
        </div>
        <script>
            const audioConstraints = {
                audio: {
                    channelCount: 1,
                    sampleRate: 16000,
                    sampleSize: 16,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            };

            let mediaRecorder;
            let audioContext;
            let audioSource;
            let audioProcessor;
            let mediaStream;
            let ws;
            let isRecording = false;

            const startButton = document.getElementById('startButton');
            const statusIndicator = document.getElementById('statusIndicator');
            const statusText = document.getElementById('statusText');
            const transcript = document.getElementById('transcript');
            const debugLog = document.getElementById('debugLog');

            function updateStatus(status, message) {
                statusIndicator.className = 'status-indicator ' + status;
                statusText.textContent = message;
                log(message);
            }

            function log(message) {
                console.log(message);
                const logEntry = document.createElement('div');
                logEntry.textContent = `${new Date().toLocaleTimeString()}: ${message}`;
                debugLog.appendChild(logEntry);
                debugLog.scrollTop = debugLog.scrollHeight;
            }
            
            async function startRecording() {
                try {
                    updateStatus('active', 'Requesting microphone access...');
                    mediaStream = await navigator.mediaDevices.getUserMedia(audioConstraints);
                    updateStatus('active', 'Microphone access granted');
                    
                    // Initialize Web Audio API
                    audioContext = new AudioContext({
                        sampleRate: 16000,
                        channelCount: 1,
                        latencyHint: 'interactive'
                    });
                    
                    audioSource = audioContext.createMediaStreamSource(mediaStream);
                    
                    // Use 512 samples to match Porcupine's frame length
                    audioProcessor = audioContext.createScriptProcessor(512, 1, 1);
                    
                    // Create WebSocket connection
                    ws = new WebSocket('ws://localhost:8000/ws');
                    
                    ws.onopen = () => {
                        updateStatus('active', 'WebSocket connected');
                        startButton.textContent = 'Stop Listening';
                        
                        // Connect audio processing
                        audioSource.connect(audioProcessor);
                        audioProcessor.connect(audioContext.destination);
                        
                        audioProcessor.onaudioprocess = (e) => {
                            if (isRecording && ws.readyState === WebSocket.OPEN) {
                                const inputData = e.inputBuffer.getChannelData(0);
                                // Convert float32 to int16
                                const samples = new Int16Array(inputData.length);
                                for (let i = 0; i < inputData.length; i++) {
                                    // Normalize and scale to 16-bit range
                                    const s = Math.max(-1, Math.min(1, inputData[i]));
                                    samples[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                                }
                                ws.send(samples.buffer);
                            }
                        };
                        
                        isRecording = true;
                        log("Recording started with 512 samples buffer size");
                    };
                    
                    ws.onclose = () => {
                        updateStatus('', 'WebSocket disconnected');
                        stopRecording();
                    };
                    
                    ws.onerror = (error) => {
                        updateStatus('error', 'WebSocket error');
                        console.error(error);
                        stopRecording();
                    };
                    
                    ws.onmessage = (event) => {
                        const message = JSON.parse(event.data);
                        if (message.wake_word_detected) {
                            updateStatus('detected', 'Wake word detected!');
                            document.body.style.backgroundColor = '#90EE90';
                            setTimeout(() => {
                                document.body.style.backgroundColor = '';
                                updateStatus('active', 'Listening...');
                            }, 1000);
                        } else if (message.type === 'transcription') {
                            transcript.textContent = message.text;
                            updateStatus('active', 'Transcription received');
                        } else if (message.type === 'error') {
                            updateStatus('error', message.message);
                        }
                    };
                    
                } catch (error) {
                    updateStatus('error', 'Error starting recording: ' + error.message);
                    console.error(error);
                }
            }
            
            function stopRecording() {
                if (isRecording) {
                    log("Stopping recording...");
                    isRecording = false;
                    
                    // Stop all audio processing
                    if (audioProcessor) {
                        audioProcessor.disconnect();
                        audioSource.disconnect();
                    }
                    
                    // Close audio context
                    if (audioContext && audioContext.state !== 'closed') {
                        audioContext.close().catch(console.error);
                    }
                    
                    // Stop all media tracks
                    if (mediaStream) {
                        mediaStream.getTracks().forEach(track => {
                            track.stop();
                            log(`Stopped media track: ${track.kind}`);
                        });
                        mediaStream = null;
                    }
                    
                    // Close WebSocket connection
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.close();
                    }
                    
                    // Reset UI
                    startButton.textContent = 'Start Listening';
                    updateStatus('', 'Microphone inactive');
                    log("Recording stopped and all resources cleaned up");
                }
            }
            
            startButton.addEventListener('click', () => {
                if (!isRecording) {
                    startRecording();
                } else {
                    stopRecording();
                }
            });

            // Cleanup on page unload
            window.addEventListener('beforeunload', () => {
                stopRecording();
            });
        </script>
    </body>
    </html>
    """

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