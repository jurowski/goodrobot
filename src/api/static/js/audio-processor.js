class AudioProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();
        // Always use 16kHz as the target sample rate
        this.targetSampleRate = 16000;
        this.sampleRate = options.processorOptions?.sampleRate || 48000;
        this.bufferSize = options.processorOptions?.bufferSize || 512;
        
        // Initialize resampling
        this.resampleRatio = this.targetSampleRate / this.inputSampleRate;
        this.buffer = new Float32Array(this.bufferSize);
        this.bufferIndex = 0;
        
        // Calculate resampling buffer size
        const resampledSize = Math.ceil(this.bufferSize * this.resampleRatio);
        this.resampleBuffer = new Float32Array(resampledSize * 4);
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
        const outputLength = Math.ceil(inputData.length * this.resampleRatio);
        const output = new Float32Array(outputLength);
        
        for (let i = 0; i < outputLength; i++) {
            const inputIndex = i / this.resampleRatio;
            const x = Math.floor(inputIndex);
            const frac = inputIndex - x;
            
            const x0 = x > 0 ? inputData[x - 1] : inputData[0];
            const x1 = inputData[x];
            const x2 = x < inputData.length - 1 ? inputData[x + 1] : x1;
            const x3 = x < inputData.length - 2 ? inputData[x + 2] : x2;
            
            const a = 0.5 * (3.0 * (x1 - x2) - x0 + x3);
            const b = x2 + x0 - 2.0 * x1;
            const c = 0.5 * (x2 - x0);
            const d = x1;
            
            output[i] = ((a * frac + b) * frac + c) * frac + d;
        }
        
        return output;
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (!input || !input.length) return true;

        const inputData = input[0];
        
        if (this.needsResampling) {
            // Add new samples to resample buffer
            inputData.forEach(sample => {
                if (this.resampleIndex < this.resampleBuffer.length) {
                    this.resampleBuffer[this.resampleIndex++] = sample;
                }
            });

            // Process when we have enough samples
            if (this.resampleIndex >= this.resampleBuffer.length / 2) {
                const resampledData = this.resample(this.resampleBuffer.slice(0, this.resampleIndex));
                this.processAudioChunk(resampledData);
                
                // Keep remaining samples
                const remainingSamples = this.resampleBuffer.slice(this.resampleIndex - this.bufferSize);
                this.resampleBuffer.fill(0);
                remainingSamples.forEach((sample, i) => this.resampleBuffer[i] = sample);
                this.resampleIndex = remainingSamples.length;
            }
        } else {
            this.processAudioChunk(inputData);
        }

        // Update metrics periodically
        if (currentTime - this.metrics.lastUpdate >= this.METRICS_UPDATE_INTERVAL / 1000) {
            const newMetrics = this.calculateMetrics(inputData);
            this.updateMetrics(newMetrics);
        }

        return true;
    }

    processAudioChunk(audioData) {
        for (let i = 0; i < audioData.length; i++) {
            if (this.bufferIndex < this.bufferSize) {
                this.buffer[this.bufferIndex++] = audioData[i];
            }

            // When buffer is full, send it
            if (this.bufferIndex >= this.bufferSize) {
                // Convert to 16-bit PCM
                const pcmData = new Int16Array(this.bufferSize);
                for (let j = 0; j < this.bufferSize; j++) {
                    const s = Math.max(-1, Math.min(1, this.buffer[j]));
                    pcmData[j] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                }

                this.port.postMessage({
                    type: 'audio',
                    data: pcmData.buffer
                }, [pcmData.buffer]);

                this.bufferIndex = 0;
            }
        }
    }

    updateMetrics(newMetrics) {
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
}

registerProcessor('audio-processor', AudioProcessor);
