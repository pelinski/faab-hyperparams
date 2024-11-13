(
var bufferLen = 512; // eq. to seq_len in python code

// Function to create OSCdef handlers
var createOSCHandler = { |name, address|
    OSCdef(
        name,
        {
            arg val;
            if (val.size == bufferLen) {
                ("OK length: " ++ val.size).postln;
            } {
                ("Received data of incorrect length: " ++ val.size).postln;
                ("First item: " ++ val[0]).postln;
                ("Last item: " ++ val[val.size - 1]).postln;
            }
        },
        address,
        recvPort: 2222
    );
};

createOSCHandler.(\getBuffer1, '/f1');
createOSCHandler.(\getBuffer2, '/f2');
createOSCHandler.(\getBuffer3, '/f3');
createOSCHandler.(\getBuffer4, '/f4');
)