let audioContext;
let audioWorkletNode;
let ws;
let isRecording = false;

const audioConstraints = {
    channelCount: 1,
    sampleRate: 16000,
    sampleSize: 16
};

async function initAudio() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                sampleRate: 16000,
                sampleSize: 16,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            }
        });
        console.log('Microphone access granted');

        audioContext = new AudioContext({
            sampleRate: 16000,
            latencyHint: 'interactive'
        });

        // Ensure the audio context is running
        if (audioContext.state === 'suspended') {
            await audioContext.resume();
        }

        // Load and register the audio worklet
        await audioContext.audioWorklet.addModule('audio-processor.js');
        console.log('Audio worklet loaded');

        const source = audioContext.createMediaStreamSource(stream);
        audioWorkletNode = new AudioWorkletNode(audioContext, 'audio-processor', {
            numberOfInputs: 1,
            numberOfOutputs: 1,
            processorOptions: {
                sampleRate: 16000,
                bufferSize: 512
            }
        });

        // Handle audio chunks from the worklet
        audioWorkletNode.port.onmessage = (event) => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(event.data);
            }
        };

        return true;
    } catch (err) {
        console.error('Error initializing audio:', err);
        return false;
    }
}

async function startRecording() {
    if (isRecording) return;

    try {
        // Initialize audio if not already done
        if (!audioContext) {
            const success = await initAudio();
            if (!success) {
                throw new Error('Failed to initialize audio');
            }
        }

        // Resume audio context if suspended
        if (audioContext.state === 'suspended') {
            await audioContext.resume();
        }

        // Setup WebSocket
        ws = new WebSocket('ws://localhost:8000/voice');

        ws.onopen = () => {
            console.log('WebSocket connected');
            // Connect audio processing chain
            const source = audioContext.createMediaStreamSource(stream);
            source.connect(audioWorkletNode);
            audioWorkletNode.connect(audioContext.destination);
            isRecording = true;
        };

        ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            if (message.type === 'transcription') {
                console.log('Transcription:', message.text);
                // Handle transcription result
            } else if (message.type === 'error') {
                console.error('Server error:', message.error);
            }
        };

        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            stopRecording();
        };

        ws.onclose = () => {
            console.log('WebSocket closed');
            stopRecording();
        };

    } catch (err) {
        console.error('Error starting recording:', err);
        stopRecording();
    }
}

function stopRecording() {
    if (!isRecording) return;

    try {
        if (audioWorkletNode) {
            audioWorkletNode.disconnect();
        }

        if (ws) {
            ws.close();
            ws = null;
        }

        isRecording = false;
        console.log('Recording stopped');
    } catch (err) {
        console.error('Error stopping recording:', err);
    }
}

// Export functions for use in HTML
window.startRecording = startRecording;
window.stopRecording = stopRecording;
