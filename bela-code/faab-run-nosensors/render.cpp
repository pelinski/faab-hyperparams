// rsync -avL bela-code/faab-run-nosensors-osc root@bela.local:Bela/projects/

#include <Bela.h>
#include <Watcher.h>
#include <vector>
#include "BinaryStreamer.h"

#define NUM_SENSORS 8

unsigned int gAudioFramesPerAnalogFrame;
float gInvSampleRate;
float gInvAudioFramesPerAnalogFrame;

float gain = 1.7f;

std::vector<Watcher<float>*> gFaabWatchers;
std::vector<BinaryStreamer*> gStreamers;
const int gBasePriorityAuxTask = 90;
const int gPlaybackBufferSize = 22050;

bool setup(BelaContext* context, void* userData) {
    gAudioFramesPerAnalogFrame = context->audioFrames / context->analogFrames;

    Bela_getDefaultWatcherManager()->getGui().setup(context->projectName);
    Bela_getDefaultWatcherManager()->setup(context->audioSampleRate);

    // Initialize Watchers for each sensor
    for (unsigned int i = 0; i < NUM_SENSORS; ++i) {
        gFaabWatchers.push_back(new Watcher<float>("gFaabSensor_" + std::to_string(i + 1)));
        int priority = gBasePriorityAuxTask - i;
        gStreamers.push_back(new BinaryStreamer(gPlaybackBufferSize, priority));
        gStreamers[i]->setUnderrunCallback([i]() { rt_printf("Warning: File %zu buffer underrun!\n", i); });

        std::string filename = "data/gFaabSensor_" + std::to_string(i + 1) + ".bin";
        if (!gStreamers[i]->loadFile(filename.c_str())) {
            printf("Failed to load file %zu: %s\n", i, filename.c_str());
            return false;
        }
        printf("Loaded file %zu: %s (%s)\n", i, filename.c_str(), gStreamers[i]->getVarName().c_str());
    }

    return true;
}

void render(BelaContext* context, void* userData) {
    for (unsigned int n = 0; n < context->audioFrames; n++) {
        uint64_t frames = context->audioFramesElapsed + n;
        Bela_getDefaultWatcherManager()->tick(frames);

        if (gAudioFramesPerAnalogFrame && !(n % gAudioFramesPerAnalogFrame)) { // analog sample rate is half than audio sample rate

            float out = 0.0f;

            for (unsigned int i = 0; i < NUM_SENSORS; i++) {
                if (gStreamers[i]->isReady()) {
                    float sample = gStreamers[i]->getNextSample();
                    *gFaabWatchers[i] = sample;
                    out += sample;
                }
            }

            audioWrite(context, n, 0, gain * out);
            audioWrite(context, n, 1, gain * out);
        }
    }
}

void cleanup(BelaContext* context, void* userData) {
}
