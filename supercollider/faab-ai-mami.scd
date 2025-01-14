(
var bufferLen = 1024; // Length of each incoming OSC buffer
var numChannels = 4; // Number of channels
var latencySamples = 2.5*1024; // Latency in samples
var sampleRate = 44100; // Assume default sample rate (change if needed)
var writePointers = Array.fill(numChannels, 0);
var readPointers = Array.fill(numChannels, { latencySamples });
var circularBufferSize = 10*1024; // Size of the circular buffer (must be a power of 2)
var circularBuffers = Array.fill(numChannels, { Array.fill(circularBufferSize, 0) });

var audioOutput = Array.fill(numChannels, 0);

// Function to handle incoming OSC messages
var createOSCHandler = { |name, address|
    OSCdef(
        name,
        {
            arg msg;
            var receivedBuffer = msg[1..];
            var id = msg[0].asString[2].digit - 1; // Extract channel ID (1-based)
            
            if (receivedBuffer.size == bufferLen) {
                var writePtr = writePointers[id];
                // Write data into the circular buffer
                bufferLen.do { |i|
                    circularBuffers[id][(writePtr + i) % circularBufferSize] = receivedBuffer[i];
                };
                // Update the write pointer
                writePointers[id] = (writePtr + bufferLen) % circularBufferSize;
            } {
                ("Received data of incorrect length: " ++ receivedBuffer.size).postln;
            }
        },
        address,
        recvPort: 2222
    );
};

// Create OSC handlers for each channel
createOSCHandler.(\getBuffer1, '/f1');
createOSCHandler.(\getBuffer2, '/f2');
createOSCHandler.(\getBuffer3, '/f3');
createOSCHandler.(\getBuffer4, '/f4');

// Synth for playback
SynthDef(\circularBufferPlayer, {
    arg out=0;
    var sig;

    sig = Array.fill(numChannels, { |channel|
        var readPtr = readPointers[channel];
        var writePtr = writePointers[channel];
        var sample;

        // Read data from the circular buffer
        sample = circularBuffers[channel][readPtr % circularBufferSize];
        readPointers[channel] = (readPtr + 1) % circularBufferSize;

        // Log underrun condition
        if (readPtr == writePtr) {
            ("Buffer underrun on channel " ++ channel).postln;
        };

        sample; // Return the sample for this channel
    });

    // Mix the signals and play through speakers
    Out.ar(out, sig);
}).add;

// Start the player
s.waitForBoot({
    Synth(\circularBufferPlayer);
});
)
