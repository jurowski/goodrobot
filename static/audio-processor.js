class AudioProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();
        // Porcupine expects 512 samples
        this.bufferSize = 512;
        this.sampleRate = 16000;
        this.buffer = new Float32Array(this.bufferSize);
        this.bufferIndex = 0;
        console.log(`AudioProcessor initialized with buffer size ${this.bufferSize} and sample rate ${this.sampleRate}`);
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
                this.port.postMessage(pcmData.buffer, [pcmData.buffer]);

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