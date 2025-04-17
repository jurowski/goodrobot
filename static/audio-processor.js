class AudioProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();
        // Porcupine expects 512 samples
        this.bufferSize = 512;
        this.sampleRate = 16000;
        this.buffer = new Float32Array(this.bufferSize);
        this.bufferIndex = 0;

        // Audio quality metrics
        this.metrics = {
            rms: 0,          // Root Mean Square (volume level)
            peak: 0,         // Peak amplitude
            clipCount: 0,    // Number of clipped samples
            silenceCount: 0, // Number of silent frames
            noiseFloor: -60, // Noise floor in dB
            snr: 0,          // Signal-to-Noise Ratio
            lastUpdate: 0    // Last metrics update timestamp
        };

        // Constants for audio quality analysis
        this.CLIPPING_THRESHOLD = 0.99;      // Threshold for detecting clipping
        this.SILENCE_THRESHOLD = 0.01;       // Threshold for detecting silence
        this.METRICS_UPDATE_INTERVAL = 100;  // Update metrics every 100ms

        console.log(`AudioProcessor initialized with buffer size ${this.bufferSize} and sample rate ${this.sampleRate}`);
    }

    calculateMetrics(inputData) {
        // Calculate RMS (volume level)
        const sumSquares = inputData.reduce((sum, sample) => sum + sample * sample, 0);
        const rms = Math.sqrt(sumSquares / inputData.length);
        
        // Calculate peak amplitude
        const peak = Math.max(...inputData.map(Math.abs));
        
        // Count clipped samples
        const clipped = inputData.filter(sample => Math.abs(sample) > this.CLIPPING_THRESHOLD).length;
        
        // Check for silence
        const isSilent = rms < this.SILENCE_THRESHOLD;
        
        // Calculate noise floor (in dB)
        const noiseFloor = 20 * Math.log10(Math.min(...inputData.map(Math.abs).filter(x => x > 0)) || 0.000001);
        
        // Calculate SNR
        const signalPower = sumSquares / inputData.length;
        const noisePower = isSilent ? signalPower : Math.pow(10, noiseFloor / 10);
        const snr = signalPower > 0 ? 10 * Math.log10(signalPower / noisePower) : 0;

        return {
            rms,
            peak,
            clipped,
            isSilent,
            noiseFloor,
            snr,
            timestamp: currentTime
        };
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (!input || !input[0]) {
            console.log("No input data available");
            return true;
        }

        const inputData = input[0];
        
        // Add incoming audio data to buffer
        for (let i = 0; i < inputData.length; i++) {
            if (this.bufferIndex < this.bufferSize) {
                this.buffer[this.bufferIndex++] = inputData[i];
            }
        }

        // Update audio quality metrics periodically
        if (currentTime - this.metrics.lastUpdate >= this.METRICS_UPDATE_INTERVAL / 1000) {
            const newMetrics = this.calculateMetrics(inputData);
            
            // Update running metrics
            this.metrics.rms = newMetrics.rms;
            this.metrics.peak = newMetrics.peak;
            this.metrics.clipCount += newMetrics.clipped;
            this.metrics.silenceCount += newMetrics.isSilent ? 1 : 0;
            this.metrics.noiseFloor = newMetrics.noiseFloor;
            this.metrics.snr = newMetrics.snr;
            this.metrics.lastUpdate = currentTime;

            // Send metrics to main thread
            this.port.postMessage({
                type: 'metrics',
                data: {
                    rms: this.metrics.rms,
                    peak: this.metrics.peak,
                    clipCount: this.metrics.clipCount,
                    silenceCount: this.metrics.silenceCount,
                    noiseFloor: this.metrics.noiseFloor,
                    snr: this.metrics.snr,
                    timestamp: currentTime
                }
            });
        }

        // If buffer is full, convert and send
        if (this.bufferIndex >= this.bufferSize) {
            try {
                // Convert to 16-bit PCM
                const pcmData = new Int16Array(this.bufferSize);
                for (let i = 0; i < this.bufferSize; i++) {
                    // Normalize and convert to 16-bit PCM
                    const s = Math.max(-1, Math.min(1, this.buffer[i]));
                    pcmData[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                }

                // Send the PCM data to the main thread
                this.port.postMessage({
                    type: 'audio',
                    data: pcmData.buffer
                }, [pcmData.buffer]);

                // Reset buffer
                this.buffer = new Float32Array(this.bufferSize);
                this.bufferIndex = 0;
            } catch (error) {
                console.error("Error processing audio data:", error);
            }
        }

        return true;
    }
}

try {
    // Register the processor
    registerProcessor('audio-processor', AudioProcessor);
    console.log('AudioProcessor registered successfully');
} catch (error) {
    console.error('Failed to register AudioProcessor:', error);
    throw error;
} 