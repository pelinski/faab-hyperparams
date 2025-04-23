(
var bufferLen = 1024; // Length of each incoming OSC buffer
var numChannels = 4; // Number of channels
var latencySamples = 2 * bufferLen; // Latency in samples
var sampleRate = 44100; // Assume default sample rate (change if needed)
var circularBufferSize = 20 * bufferLen; // Size of the circular buffer (must be a power of 2)
var writePointers = Array.fill(numChannels, {latencySamples});
var readPointers = Array.fill(numChannels, { latencySamples });
var intMBuffer = 2048.collect{};

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

/*s.options.outDevice_(

	"Fireface UCX (23732049)"

);

s.options.inDevice_(
	"Fireface UCX (23732049)"

);*/

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

				receivedBuffer.do{|val, idx|
					var index = idx * 2;

					2.do{|n|
						var realIndex = index + n;
						intMBuffer[realIndex] = val
					};
				};

				if (receivedBuffer.size == bufferLen) {
					var writePtr = writePointers[id];
					circularBuffers[id].setn(writePtr, intMBuffer);
					writePointers[id] = (writePtr + (2 * bufferLen)) % circularBufferSize;
				} {
					("Received data of incorrect length: " ++ receivedBuffer.size).postln;
				}
			},
			address,
			recvPort: 2222
		);
	};

	~modelBusses = numChannels.collect{Bus.audio(s)};

	~inGrp = Group.new;
	~audioOutGrp = Group.new(~inGrp, \addAfter);

	s.sync;

	// Create OSC handlers for each channel
	createOSCHandler.(\getBuffer1, '/f1');
	createOSCHandler.(\getBuffer2, '/f2');
	createOSCHandler.(\getBuffer3, '/f3');
	createOSCHandler.(\getBuffer4, '/f4');

	// SynthDef to play the audio


	SynthDef(\circularBufferPlayer, {|rocTSamp = 2, out = 0, modelBus|
		var sig, faabIn;

		faabIn = SoundIn.ar([0, 1]).scope;
		sig = Array.fill(numChannels, { |channel|
			//var readPtr = LocalIn.kr(1); // Use a local control bus to manage pointers
			/*var readPtr = readPointers[channel];
			var writePtr = writePointers[channel];*/
			// var readPtr = Line.ar(0, circularBufferSize, circularBufferSize * SampleDur.ir);
			// var writePtr = writePointers[channel];
			var sample, roc;

			sample = PlayBuf.ar(1, circularBuffers[channel], BufRateScale.kr(circularBuffers[channel]), loop: 1) * 0.1;

			sample = LeakDC.ar(sample);
			/*sample = LinLin.ar(sample, RunningMin.ar(sample, DelayN.ar(Impulse), RunningMax.ar(sample));*/
			// sample = (sample - 0.5) * 2;
			// sample = sample * 0.5 + 0.5;
			Out.ar(~modelBusses[channel], sample.scope);
			sample;
		});

		Out.ar(2, sig); // Mix channels and scale down the output
		Out.ar(6, faabIn);
	}).add;

	SynthDef(\complexosc, {|freq1 = 40, freq2 = 0.5, fm1 = 0.1, fm2 = 0.1, pm1 = 0, pm2 = 0, mul1 = 1, mul2 = 1, ring1 = 1, ring2 = 1, add1 = 0, add2 = 0, fold = 0.8, wrap = 0.9, switch = 0, offSet = 1, lpFreq = 0.9, modelBus|
		var modelBusIn, osc1=1, osc2=1, fbk, outlet, folded, wrapped, runningMax;
		modelBusIn = In.ar(modelBus, numChannels).scope;
		runningMax = RunningMax.ar(modelBusIn);
		// modelBusIn = (modelBusIn - OnePole.ar(modelBusIn, lpFreq));
		/*		fm2 = modelBusIn[2];*/
		// pm2 = modelBusIn[3];
		fbk = LocalIn.ar(2,0);
		osc1 = SinOsc.ar(freq1 + (fbk[1].range(1,1000) * fm1), 0+(fbk[1].range(-pi,pi)*pm1), mul1+(fbk[1]*ring1), add1);
		osc2 = SinOsc.ar(freq2 * (offSet + modelBusIn[0]).poll + (fbk[0].range(1,1000) * fm2) , 0+(fbk[1].range(-pi,pi)*pm2) * (offSet + modelBusIn[1].scope), mul2+(fbk[0]*ring2), add2);
		folded = Fold.ar(osc1!2, fold.neg,fold);
		wrapped = Wrap.ar(osc1!2,wrap.neg,wrap);
		outlet = LeakDC.ar(Select.ar(switch, [osc1!2, osc2!2, folded, wrapped]));
		LocalOut.ar([osc1, osc2]);
		Out.ar(0, outlet * 0.1)
	}
	).add;

	// Start playback
	/*s.waitForBoot({
	/*	circularBuffers.do(_.zero); // Clear buffers*/
	Synth(\circularBufferPlayer);
	});*/
	s.sync;
	x = Synth(\circularBufferPlayer, target: ~inGrp);
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

