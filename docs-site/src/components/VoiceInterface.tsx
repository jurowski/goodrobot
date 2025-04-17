import React, { useState, useEffect, useRef } from 'react';
import styles from '../styles/VoiceInterface.module.css';

interface VoiceInterfaceProps {
  apiUrl?: string;
}

const VoiceInterface: React.FC<VoiceInterfaceProps> = ({ apiUrl = 'http://localhost:8000' }) => {
  const [isListening, setIsListening] = useState(false);
  const [transcription, setTranscription] = useState('');
  const [rawTranscription, setRawTranscription] = useState('');
  const [response, setResponse] = useState('');
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const [voiceProvider, setVoiceProvider] = useState('web-speech');
  const [apiKey, setApiKey] = useState('');
  const [status, setStatus] = useState('Microphone inactive');
  const [volume, setVolume] = useState(0);
  const [noiseFloor, setNoiseFloor] = useState(0);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  useEffect(() => {
    // Initialize audio context and analyser
    const initAudio = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioContextRef.current = new AudioContext();
        analyserRef.current = audioContextRef.current.createAnalyser();
        const source = audioContextRef.current.createMediaStreamSource(stream);
        source.connect(analyserRef.current);
        analyserRef.current.fftSize = 256;
        
        // Start volume visualization
        visualizeVolume();
      } catch (error) {
        console.error('Error initializing audio:', error);
        setStatus('Error initializing audio');
      }
    };

    initAudio();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, []);

  const visualizeVolume = () => {
    if (!analyserRef.current) return;

    const bufferLength = analyserRef.current.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    const updateVolume = () => {
      analyserRef.current?.getByteFrequencyData(dataArray);
      
      // Calculate average volume
      const average = dataArray.reduce((a, b) => a + b) / bufferLength;
      setVolume(average);
      
      // Update noise floor (minimum volume)
      setNoiseFloor(Math.min(...dataArray));
      
      animationFrameRef.current = requestAnimationFrame(updateVolume);
    };
    
    updateVolume();
  };

  const startListening = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      const chunks: Blob[] = [];

      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunks.push(e.data);
        }
      };

      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(chunks, { type: 'audio/wav' });
        await processAudio(audioBlob);
      };

      mediaRecorderRef.current.start();
      setIsListening(true);
      setStatus('Listening...');
    } catch (error) {
      console.error('Error starting recording:', error);
      setStatus('Error starting recording');
    }
  };

  const stopListening = () => {
    if (mediaRecorderRef.current && isListening) {
      mediaRecorderRef.current.stop();
      setIsListening(false);
      setStatus('Processing...');
    }
  };

  const processAudio = async (audioBlob: Blob) => {
    try {
      const formData = new FormData();
      formData.append('audio', audioBlob);
      formData.append('context', JSON.stringify({
        voiceEnabled,
        voiceProvider,
        apiKey
      }));

      const response = await fetch(`${apiUrl}/voice`, {
        method: 'POST',
        body: formData
      });

      const data = await response.json();
      setTranscription(data.transcription || '');
      setRawTranscription(data.rawTranscription || '');
      setResponse(data.response || '');
      setStatus('Ready');
    } catch (error) {
      console.error('Error processing audio:', error);
      setStatus('Error processing audio');
    }
  };

  const handleTextInput = async (text: string) => {
    try {
      setStatus('Processing...');
      const response = await fetch(`${apiUrl}/voice`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          command: text,
          context: {
            voiceEnabled,
            voiceProvider,
            apiKey
          }
        })
      });

      const data = await response.json();
      setTranscription(text);
      setRawTranscription(text);
      setResponse(data.response || '');
      setStatus('Ready');
    } catch (error) {
      console.error('Error processing text:', error);
      setStatus('Error processing text');
    }
  };

  return (
    <div className={styles.container}>
      <div className={styles.panel}>
        <h2>Voice & Text AI Assistant</h2>
        <p>Interact with the assistant through voice or text. Say "Jarvis" to activate voice recognition, or use the text input below.</p>
        
        <div className={styles.controlsRow}>
          <div className={styles.statusGroup}>
            <span className={styles.statusIndicator} style={{ backgroundColor: isListening ? '#4CAF50' : '#FF5252' }} />
            <span>{status}</span>
          </div>
          <button 
            className={styles.controlButton} 
            onClick={startListening}
            disabled={isListening}
          >
            Start Listening
          </button>
          <button 
            className={styles.controlButton} 
            onClick={stopListening}
            disabled={!isListening}
          >
            Done Speaking
          </button>
        </div>

        <div className={styles.visualizerContainer}>
          <div 
            className={styles.volumeBar} 
            style={{ height: `${volume}%` }} 
          />
          <div 
            className={styles.noiseFloor} 
            style={{ bottom: `${noiseFloor}%` }} 
          />
        </div>

        <div className={styles.voiceSettings}>
          <h3>Voice & Text Settings</h3>
          <div className={styles.voiceControls}>
            <select 
              className={styles.voiceSelect}
              value={voiceProvider}
              onChange={(e) => setVoiceProvider(e.target.value)}
            >
              <option value="web-speech">Web Speech (Free)</option>
              <option value="elevenlabs">ElevenLabs</option>
            </select>
            <div className={styles.voiceToggle}>
              <input 
                type="checkbox" 
                id="voiceEnabled"
                checked={voiceEnabled}
                onChange={(e) => setVoiceEnabled(e.target.checked)}
              />
              <label htmlFor="voiceEnabled">Enable Voice</label>
            </div>
          </div>
          {voiceProvider === 'elevenlabs' && (
            <input
              type="password"
              className={styles.apiKeyInput}
              placeholder="ElevenLabs API Key"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
            />
          )}
        </div>

        <div className={styles.transcriptionContainer}>
          <div className={styles.transcriptionLabel}>Transcription</div>
          <div className={styles.transcriptionText}>{transcription}</div>
          <div className={styles.transcriptionLabel}>Raw Transcription</div>
          <div className={styles.transcriptionText}>{rawTranscription}</div>
          
          <div className={styles.textInputContainer}>
            <div className={styles.transcriptionLabel}>
              Voice Input (Simulated Speech)
            </div>
            <div className={styles.textInputWrapper}>
              <textarea
                className={styles.textInput}
                placeholder="Type here to simulate speaking..."
                rows={3}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleTextInput(e.currentTarget.value);
                    e.currentTarget.value = '';
                  }
                }}
              />
              <button 
                className={styles.controlButton}
                onClick={() => {
                  const textarea = document.querySelector(`.${styles.textInput}`) as HTMLTextAreaElement;
                  if (textarea) {
                    handleTextInput(textarea.value);
                    textarea.value = '';
                  }
                }}
              >
                Send as Voice
              </button>
            </div>
          </div>
        </div>

        <div className={styles.responseContainer}>
          <div className={styles.responseLabel}>Response</div>
          <div className={styles.responseText}>{response}</div>
        </div>
      </div>
    </div>
  );
};

export default VoiceInterface; 