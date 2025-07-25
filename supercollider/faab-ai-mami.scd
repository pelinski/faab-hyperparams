(


s.options.memSize_(2.pow(20));
s.options.sampleRate_(sampleRate);
s.options.numInputBusChannels_(8);
s.options.numOutputBusChannels_(8);
s.options.recChannels_(8);
// s.options.hardwareBufferSize = 64;
s.options.maxSynthDefs = 2048;
s.options.maxNodes = 2048;
s.recHeaderFormat = "wav";
s.options.numBuffers = 4096 * 2;

/*s.options.outDevice_("Fireface UCX (23732049)");
s.options.inDevice_("Fireface UCX (23732049)");*/

var bufferLen = 1024; // Length of each incoming OSC buffer
var numChannels = 4; // Number of channels
var latencySamples = 4 * 2 * bufferLen; // Latency in samples
var sampleRate = 44100; // Assume default sample rate (change if needed)


var circularBufferSize = 5 * bufferLen; // Size of the circular buffer (must be a power of 2)
var writePointers = Array.fill(numChannels, {latencySamples});
var readPointers = Array.fill(numChannels, {latencySamples });


s.waitForBoot({
	// osc to update the readPointer array
	OSCdef(\readPosListener, { |msg|
		var channel = msg[2];
		var position = msg[3];
		// Update our readPointers array
		readPointers[channel] = position;
	}, '/readPos');

	var circularBuffers = Array.fill(numChannels, { Buffer.alloc(s, circularBufferSize)});
	// OSC message handler function:
	var createOSCHandler = { |name, address|
		OSCdef(
			name,
			{
				arg msg;
				var receivedBuffer = msg[1..];
				var id = msg[0].asString[2].digit - 1; // Extract channel ID (1-based)

				var intMBuffer = 2048.collect{};
				receivedBuffer.do{|val, idx|
					var index = idx * 2;

					// buffers sent from python are 1024 samples at 22050, we upsample by repeating each sample twice so that now the buffers are 2048 at 44100
					2.do{|n|
						var realIndex = index + n;
						intMBuffer[realIndex] = val
					};
				};

				if (receivedBuffer.size == bufferLen) {
					var writePtr = writePointers[id];
					var readPtr = readPointers[id];
                // Account for wraparound when comparing pointers
                   var distance = if(readPtr <= writePtr,
                    { readPtr + circularBufferSize - writePtr },
                    { readPtr - writePtr });
            // Check if write operation is safe
                // Ensure we don't write over data that hasn't been read yet
                if (distance > intMBuffer.size) {
                    circularBuffers[id].setn(writePtr, intMBuffer);
                    writePointers[id] = (writePtr + intMBuffer.size) % circularBufferSize;
                } {
                    ("Warning: Write pointer too close to read pointer on channel " ++ id).postln;
                    // Could implement a strategy here - perhaps wait or skip
                };
            } {
                ("Received data of incorrect length: " ++ receivedBuffer.size).postln;
            }
        },
        address,
        recvPort: 2222
    );
};
And for your player synth:

supercollider
SynthDef(\circularBufferPlayer, {|out = 0, modelBus|
    var sig, mixedSig;

    sig = Array.fill(numChannels, { |channel|
        var readPtr, sample;

        // Create advancing read pointer with proper wraparound
        readPtr = Phasor.ar(
            0,                   // No trigger (free running)
            1,                   // Advance 1 sample per sample
            0,                   // Start of buffer (for wraparound)
            circularBufferSize,  // End of buffer (for wraparound)
            readPointers[channel] // Initial position
        );

        // Read from the buffer at calculated position
        sample = BufRd.ar(
            1,                    // Mono channel
            circularBuffers[channel],
            readPtr,              // Position to read from
            1,                    // Linear interpolation
            1                     // Wrap around buffer bounds
        ) * 0.1;

        // Update read pointer position via OSC
        SendReply.kr(Impulse.kr(30), '/readPos', readPtr, channel);

        sample = LeakDC.ar(sample);
        Out.ar(~modelBusses[channel], sample);
        sample;
    });

    mixedSig = sig.sum;
    Out.ar([0, 1], mixedSig);
}).add;

// Add OSC handler to update readPointers
OSCdef(\readPosListener, { |msg|
    var channel = msg[2];
    var position = msg[3];
    readPointers[channel] = position;
}, '/readPos');
This implementation:

Calculates the distance between read and write pointers, accounting for buffer wraparound
Ensures there's enough space before writing new data
Updates the read pointer values through an OSC handler
Correctly handles the upsampled buffer sizes in your write operation
The key point is simply ensuring that the distance between the read and write pointers is always greater than the size of the data you're writing to avoid overwriting data that hasn't been read yet.







	// Create OSC handlers for each channel
	createOSCHandler.(\getBuffer1, '/f1');
	createOSCHandler.(\getBuffer2, '/f2');
	createOSCHandler.(\getBuffer3, '/f3');
	createOSCHandler.(\getBuffer4, '/f4');


	~modelBusses = numChannels.collect{Bus.audio(s)};
	~inGrp = Group.new;
	~audioOutGrp = Group.new(~inGrp, \addAfter);

	s.sync;


	// SynthDef to play the audio
	SynthDef(\circularBufferPlayer, {|rocTSamp = 2, out = 0, modelBus|

		var sig, mixedSig;
		sig = Array.fill(numChannels, { |channel|
			var readPtr, sample;


			// continuously incrementing read pointer
			// Create a continuously incrementing read pointer
			readPtr = Phasor.ar(
				0,                          // trigger: 0 means free-running
				1,                          // rate: increment by 1 sample per sample
				0,                          // start position
				circularBufferSize,         // end position
				readPointers[channel]       // initial position
			);

			// Read from buffer at the calculated position
			sample = BufRd.ar(
				1,                          // number of channels
				circularBuffers[channel],   // buffer number
				readPtr,                    // read position
				1,                          // interpolation (1 = linear)
				1                           // wrap (1 = yes)
			) * 0.1;

			// Send the current read position back to keep readPointers updated
			SendReply.kr(Impulse.kr(30), '/readPos', readPtr, channel);


			//var readPtr = LocalIn.kr(1); // Use a local control bus to manage pointers
			/*var readPtr = readPointers[channel];
			var writePtr = writePointers[channel];*/
			// var readPtr = Line.ar(0, circularBufferSize, circularBufferSize * SampleDur.ir);
			// var writePtr = writePointers[channel];

			// sample = PlayBuf.ar(1, circularBuffers[channel], BufRateScale.kr(circularBuffers[channel]), loop: 1) * 0.1;
			sample = LeakDC.ar(sample);
			/*sample = LinLin.ar(sample, RunningMin.ar(sample, DelayN.ar(Impulse), RunningMax.ar(sample));*/
			// sample = (sample - 0.5) * 2;
			// sample = sample * 0.5 + 0.5;
			Out.ar(~modelBusses[channel], sample.scope);
			sample;
		});

		mixedSig = sig.sum; // Mix channels
		Out.ar([0, 1], mixedSig); // Send to output channels 0 and 1

		// // play faab in (ch. 0 and 1) ?
		// var faabIn:
		// faabIn = SoundIn.ar([0, 1]).scope;
		// Out.ar(6, faabIn);
	}).add;

	// SynthDef(\complexosc, {|freq1 = 40, freq2 = 0.5, fm1 = 0.1, fm2 = 0.1, pm1 = 0, pm2 = 0, mul1 = 1, mul2 = 1, ring1 = 1, ring2 = 1, add1 = 0, add2 = 0, fold = 0.8, wrap = 0.9, switch = 0, offSet = 1, lpFreq = 0.9, modelBus|
	// 	var modelBusIn, osc1=1, osc2=1, fbk, outlet, folded, wrapped, runningMax;
	// 	modelBusIn = In.ar(modelBus, numChannels).scope;
	// 	runningMax = RunningMax.ar(modelBusIn);
	// 	// modelBusIn = (modelBusIn - OnePole.ar(modelBusIn, lpFreq));˘
	// 	/*		fm2 = modelBusIn[2];*/
	// 	// pm2 = modelBusIn[3];
	// 	fbk = LocalIn.ar(2,0);
	// 	osc1 = SinOsc.ar(freq1 + (fbk[1].range(1,1000) * fm1), 0+(fbk[1].range(-pi,pi)*pm1), mul1+(fbk[1]*ring1), add1);
	// 	osc2 = SinOsc.ar(freq2 * (offSet + modelBusIn[0]).poll + (fbk[0].range(1,1000) * fm2) , 0+(fbk[1].range(-pi,pi)*pm2) * (offSet + modelBusIn[1].scope), mul2+(fbk[0]*ring2), add2);
	// 	folded = Fold.ar(osc1!2, fold.neg,fold);
	// 	wrapped = Wrap.ar(osc1!2,wrap.neg,wrap);
	// 	outlet = LeakDC.ar(Select.ar(switch, [osc1!2, osc2!2, folded, wrapped]));
	// 	LocalOut.ar([osc1, osc2]);
	// 	Out.ar(0, outlet * 0.1)
	// 	}
	// ).add;

	// Start playback
	/*s.waitForBoot({
	/*	circularBuffers.do(_.zero); // Clear buffers*/
	Synth(\circularBufferPlayer);
	});*/
	/*	s.sync;
	x = Synth(\circularBufferPlayer, target: ~inGrp);*/
	// y = Synth(\complexosc, [\modelBus, ~modelBusses], target: ~audioOutGrp);
});
)

// y.free;
// y.set(\switch, 0);
// y.set(\freq2, 1.5);
// y.set(\offSet, 1 );
// y.set(\pm1, 0.2);
// y.set(\lpFreq, 0.999);


s.plotTree;
s.scope;

