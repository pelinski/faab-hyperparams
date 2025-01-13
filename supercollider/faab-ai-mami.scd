(
s.options.memSize_(2.pow(20));
s.options.sampleRate_(44100);
s.options.numInputBusChannels_(8);
s.options.numOutputBusChannels_(8);
s.options.recChannels_(6);
s.options.hardwareBufferSize = 64;
s.options.maxSynthDefs = 2048;
s.options.maxNodes = 2048;
s.recHeaderFormat = "wav";
s.options.numBuffers = 4096 * 2;

s.waitForBoot({
    var createOSCHandler;
    var bufferLen = 512;
    var synthDef;
    
    // Define the SynthDef once, outside the OSC handler
    SynthDef(\playBuffer, { |out=0, bufnum|
		var sig = PlayBuf.ar(1, bufnum, (BufRateScale.kr(bufnum) / 2).asInteger, doneAction: 2) * 0.25;
		// var audioRate = A2K.ar(out);
		Out.ar(0, sig);
    }).add;
    
    s.sync;
    
    createOSCHandler = { |name, address|
        OSCdef(name, { |msg|
            var receivedBuffer = msg[1..];
            var id = msg[0].asString[2].digit - 1;
            
            if(receivedBuffer.size == bufferLen, {
                // Create buffer and load data
                Buffer.loadCollection(s, receivedBuffer, 1, action: { |buf|
                    // Schedule buffer playback
                    fork {
                        // Play the buffer
                        Synth(\playBuffer, [\bufnum, buf.bufnum]);
                        
                        // Wait for slightly longer than the buffer duration to ensure playback completes
                        (bufferLen/s.sampleRate + 0.1).wait;
                        
                        // Free the buffer
                        buf.free;
                        
                        ("Buffer % played and freed".format(id)).postln;
                    };
                });
            }, {
                "Received incorrect buffer size: %".format(receivedBuffer.size).warn;
            });
        }, address, recvPort: 2222);
    };
    
    // Create handlers for all four buffers
    createOSCHandler.(\getBuffer1, '/f1');
    createOSCHandler.(\getBuffer2, '/f2');
    createOSCHandler.(\getBuffer3, '/f3');
    createOSCHandler.(\getBuffer4, '/f4');
});
)