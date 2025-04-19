class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.buffer = new Float32Array(2048);
        this.bufferIndex = 0;
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (input.length > 0) {
            const inputData = input[0];
            
            // Add incoming audio data to buffer
            for (let i = 0; i < inputData.length; i++) {
                this.buffer[this.bufferIndex++] = inputData[i];
                
                // When buffer is full, process and send
                if (this.bufferIndex === this.buffer.length) {
                    // Convert to 16-bit PCM
                    const pcmData = new Int16Array(this.buffer.length);
                    for (let j = 0; j < this.buffer.length; j++) {
                        pcmData[j] = Math.max(-32768, Math.min(32767, this.buffer[j] * 32768));
                    }
                    
                    // Send the PCM data to the main thread
                    this.port.postMessage({
                        type: 'audio',
                        data: pcmData.buffer
                    }, [pcmData.buffer]);
                    
                    // Reset buffer
                    this.bufferIndex = 0;
                }
            }
        }
        
        return true;
    }
}

registerProcessor('audio-processor', AudioProcessor); 