(
var bufferLen = 1024; // Length of each incoming OSC buffer
var numChannels = 4; // Number of channels
var latencySamples = 5*1024; // Latency in samples
var sampleRate = 44100; // Assume default sample rate (change if needed)
var circularBufferSize = 10*1024; // Size of the circular buffer (must be a power of 2)

var writePointers = Array.fill(numChannels, {0});
var readPointers = Array.fill(numChannels, { latencySamples });

s.waitForBoot({
var circularBuffers = Array.fill(numChannels, { Buffer.alloc(s, circularBufferSize)});


// OSC message handler function
var createOSCHandler = { |name, address|
    OSCdef(
        name,
        {
            arg msg;
            var receivedBuffer = msg[1..];
            var id = msg[0].asString[2].digit - 1; // Extract channel ID (1-based)

            if (receivedBuffer.size == bufferLen) {
                var writePtr = writePointers[id];
                circularBuffers[id].setn(writePtr, receivedBuffer);
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

// SynthDef to play the audio
SynthDef(\circularBufferPlayer, {
    arg out = 0;
    var sig;

    sig = Array.fill(numChannels, { |channel|
        //var readPtr = LocalIn.kr(1); // Use a local control bus to manage pointers
		/*var readPtr = readPointers[channel];
        var writePtr = writePointers[channel];*/
			// var readPtr = Line.ar(0, circularBufferSize, circularBufferSize * SampleDur.ir);
		// var writePtr = writePointers[channel];
		var sample;

			sample = PlayBuf.ar(1, circularBuffers[channel], BufRateScale.kr(circularBuffers[channel]), 1, loop: 1).scope;
			// sample = BufRd.ar(1, circularBuffers[channel], readPtr, loop: 1);
        // Log underrun condition
   /*     if (readPtr == writePtr) {
            ("Buffer underrun on channel " ++ channel).postln;
        };*/
        // Increment read pointer
        //LocalOut.kr((readPtr + 1) % circularBufferSize);
        sample;
    });

    Out.ar(out, Mix(sig) * 0.1); // Mix channels and scale down the output
}).add;

// Start playback
/*s.waitForBoot({
/*	circularBuffers.do(_.zero); // Clear buffers*/
    Synth(\circularBufferPlayer);
});*/
});
)


    Synth(\circularBufferPlayer);

