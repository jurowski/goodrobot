class AudioProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();
        this.targetSampleRate = options.processorOptions.targetSampleRate || 16000;
        this.bufferSize = options.processorOptions.bufferSize || 512;
        this.buffer = new Float32Array(this.bufferSize);
        this.bufferIndex = 0;
        this.inputSampleRate = options.processorOptions.sampleRate;
        this.resampleRatio = this.inputSampleRate / this.targetSampleRate;
        this.lastInputSample = 0;
        this.resampleBuffer = new Float32Array(this.bufferSize * 3); // Larger buffer for resampling
        this.resampleIndex = 0;

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

        console.log(`AudioProcessor initialized with input rate ${this.inputSampleRate}Hz, target rate ${this.targetSampleRate}Hz, and buffer size ${this.bufferSize}`);
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

    // Improved resampling using cubic interpolation
    resample(inputData) {
        const outputLength = Math.floor(inputData.length / this.resampleRatio);
        const output = new Float32Array(outputLength);

        for (let i = 0; i < outputLength; i++) {
            const inputIndex = i * this.resampleRatio;
            const intIndex = Math.floor(inputIndex);
            const fraction = inputIndex - intIndex;

            // Get the four nearest samples for cubic interpolation
            const y0 = intIndex > 0 ? inputData[intIndex - 1] : 0;
            const y1 = inputData[intIndex];
            const y2 = intIndex < inputData.length - 1 ? inputData[intIndex + 1] : 0;
            const y3 = intIndex < inputData.length - 2 ? inputData[intIndex + 2] : 0;

            // Cubic interpolation
            const a0 = y3 - y2 - y0 + y1;
            const a1 = y0 - y1 - a0;
            const a2 = y2 - y0;
            const a3 = y1;

            const x = fraction;
            output[i] = a0 * x * x * x + a1 * x * x + a2 * x + a3;
        }

        return output;
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (!input || !input.length) return true;

        const inputData = input[0];
        
        // Add incoming data to resample buffer
        for (let i = 0; i < inputData.length; i++) {
            this.resampleBuffer[this.resampleIndex++] = inputData[i];
        }

        // When we have enough samples, resample and process
        if (this.resampleIndex >= this.bufferSize * this.resampleRatio) {
            // Resample the audio
            const resampledData = this.resample(this.resampleBuffer.slice(0, this.resampleIndex));
            
            // Process resampled data in chunks
            for (let i = 0; i < resampledData.length; i++) {
                if (this.bufferIndex < this.bufferSize) {
                    this.buffer[this.bufferIndex++] = resampledData[i];
                }

                // When buffer is full, send it
                if (this.bufferIndex >= this.bufferSize) {
                    // Convert to 16-bit PCM at target sample rate
                    const pcmData = new Int16Array(this.bufferSize);
                    for (let i = 0; i < this.bufferSize; i++) {
                        const s = Math.max(-1, Math.min(1, this.buffer[i]));
                        pcmData[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                    }

                    this.port.postMessage({
                        type: 'audio',
                        data: pcmData.buffer
                    }, [pcmData.buffer]);

                    this.bufferIndex = 0;
                }
            }

            // Reset resample buffer
            this.resampleIndex = 0;
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

        return true;
    }
}

registerProcessor('audio-processor', AudioProcessor); 