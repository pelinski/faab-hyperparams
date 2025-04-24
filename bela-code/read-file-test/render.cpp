#include <Bela.h>
#include <vector>
#include <fstream>
#include <string>
#include <BufferLoader.h>

std::vector<float> playbackBuffer;
unsigned int playbackIndex = 0;

unsigned int gAudioFramesPerAnalogFrame; // Number of audio frames per analog
                                         // frame
float gInvSampleRate;                    // 1/sample rate
float gInvAudioFramesPerAnalogFrame;     // 1/audio frames per analog frame

size_t gBufferSize = 1024; // Size of the buffer to read from file
float gain = 1.5;

bool setup(BelaContext* context, void* userData) {

    playbackBuffer = loadBufferData("gFaabSensor_5.bin", gBufferSize);

    // remove last 300 samples
    if (playbackBuffer.size() > 300) {
        playbackBuffer.resize(playbackBuffer.size() - 300);
    }

    rt_printf("Read %zu float samples from file\n", playbackBuffer.size());

    gAudioFramesPerAnalogFrame = context->audioFrames / context->analogFrames;
    gInvAudioFramesPerAnalogFrame = 1.0f / gAudioFramesPerAnalogFrame;
    gInvSampleRate = 1.0f / context->audioSampleRate;

    return true;
}

void render(BelaContext* context, void* userData) {
    for (unsigned int n = 0; n < context->audioFrames; ++n) {
        if (gAudioFramesPerAnalogFrame && !(n % gAudioFramesPerAnalogFrame)) {
            if (playbackIndex++ >= playbackBuffer.size()) {
                playbackIndex = 0; // Loop the playback
            }
            audioWrite(context, n, 0, gain * playbackBuffer[playbackIndex]);
            audioWrite(context, n, 1, gain * playbackBuffer[playbackIndex]);
        }
    }
}

void cleanup(BelaContext* context, void* userData) {
}