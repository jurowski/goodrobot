class AudioProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();
        const processorOptions = options.processorOptions || {};
        
        // Get sample rates from options or use defaults
        this.inputSampleRate = processorOptions.inputSampleRate || 48000;
        this.targetSampleRate = processorOptions.targetSampleRate || 16000;
        this.bufferSize = processorOptions.bufferSize || 512;
        
        // Calculate resampling ratio
        this.ratio = this.targetSampleRate / this.inputSampleRate;
        
        // Buffer for resampling
        this.resampleBuffer = new Float32Array(this.bufferSize);
        this.inputBuffer = new Float32Array(this.bufferSize * 4); // Larger buffer for cubic interpolation
        this.inputBufferFill = 0;
        
        // Audio metrics
        this.sampleCount = 0;
        this.rmsSum = 0;
        this.peakAmplitude = 0;
        this.clippedSamples = 0;
        this.silenceCount = 0;
        this.noiseFloor = 0.01;
        this.snr = 0;
        
        console.log(`AudioProcessor initialized with input rate: ${this.inputSampleRate}Hz, target rate: ${this.targetSampleRate}Hz, buffer size: ${this.bufferSize}`);
    }

    cubicInterpolate(y0, y1, y2, y3, mu) {
        const mu2 = mu * mu;
        const a0 = y3 - y2 - y0 + y1;
        const a1 = y0 - y1 - a0;
        const a2 = y2 - y0;
        const a3 = y1;
        
        return a0 * mu * mu2 + a1 * mu2 + a2 * mu + a3;
    }

    resample(inputData) {
        // Add new data to input buffer
        this.inputBuffer.set(inputData, this.inputBufferFill);
        this.inputBufferFill += inputData.length;
        
        // Only process if we have enough data
        if (this.inputBufferFill < 4) {
            console.log(`Not enough samples for resampling (${this.inputBufferFill}/4 needed)`);
            return null;
        }
        
        const outputLength = Math.floor(inputData.length * this.ratio);
        const output = new Float32Array(outputLength);
        
        // Calculate audio metrics
        let rms = 0;
        let peak = 0;
        let clipped = 0;
        let silence = 0;
        
        for (let i = 0; i < inputData.length; i++) {
            const sample = inputData[i];
            rms += sample * sample;
            peak = Math.max(peak, Math.abs(sample));
            if (Math.abs(sample) > 0.99) clipped++;
            if (Math.abs(sample) < this.noiseFloor) silence++;
        }
        
        rms = Math.sqrt(rms / inputData.length);
        this.rmsSum += rms;
        this.sampleCount++;
        this.peakAmplitude = Math.max(this.peakAmplitude, peak);
        this.clippedSamples += clipped;
        this.silenceCount += silence;
        
        // Calculate SNR every 100 samples
        if (this.sampleCount % 100 === 0) {
            const avgRms = this.rmsSum / 100;
            this.snr = 20 * Math.log10(avgRms / this.noiseFloor);
            this.rmsSum = 0;
            
            console.log(`Audio metrics - RMS: ${avgRms.toFixed(4)}, Peak: ${this.peakAmplitude.toFixed(4)}, Clipped: ${this.clippedSamples}, Silence: ${this.silenceCount}, SNR: ${this.snr.toFixed(2)}dB`);
            this.peakAmplitude = 0;
            this.clippedSamples = 0;
            this.silenceCount = 0;
        }
        
        // Perform resampling
        for (let i = 0; i < outputLength; i++) {
            const inputIndex = i / this.ratio;
            const index = Math.floor(inputIndex);
            
            // Ensure we have enough points for cubic interpolation
            if (index < 1 || index >= this.inputBufferFill - 2) {
                continue;
            }
            
            const fraction = inputIndex - index;
            output[i] = this.cubicInterpolate(
                this.inputBuffer[index - 1],
                this.inputBuffer[index],
                this.inputBuffer[index + 1],
                this.inputBuffer[index + 2],
                fraction
            );
        }
        
        // Keep last few samples for next interpolation
        const keepSamples = 3;
        this.inputBuffer.copyWithin(0, this.inputBufferFill - keepSamples);
        this.inputBufferFill = keepSamples;
        
        return output;
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        const output = outputs[0];
        
        if (!input || !input[0] || input[0].length === 0) {
            console.log('No input data received');
            return true;
        }
        
        // Get mono input
        const inputData = input[0];
        
        // Resample the data
        const resampled = this.resample(inputData);
        
        if (resampled) {
            // Send resampled data to main thread
            this.port.postMessage({
                type: 'audio',
                data: resampled,
                metrics: {
                    rms: this.rmsSum / this.sampleCount,
                    peak: this.peakAmplitude,
                    clipped: this.clippedSamples,
                    silence: this.silenceCount,
                    snr: this.snr
                }
            });
        }
        
        // Copy input to output (pass-through)
        output[0].set(inputData);
        
        return true;
    }
}

registerProcessor('audio-processor', AudioProcessor); 