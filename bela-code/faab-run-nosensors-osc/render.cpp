// rsync -avL bela-code/faab-run-nosensors-osc root@bela.local:Bela/projects/

#include <Bela.h>
#include <Watcher.h>
#include <vector>
#include <BufferLoader.h>

#define NUM_SENSORS 8

unsigned int gAudioFramesPerAnalogFrame;
float gInvSampleRate;
float gInvAudioFramesPerAnalogFrame;


size_t gBufferSize = 1024;
size_t gSensorPlaybackSize = 0;
float gain = 1.7f;

std::vector<Watcher<float>*> gFaabWatchers;
std::vector<float> gSensorPlayback[NUM_SENSORS];
unsigned int gPlaybackIndex = 0;

bool setup(BelaContext* context, void* userData) {
    gAudioFramesPerAnalogFrame = context->audioFrames / context->analogFrames;

    Bela_getDefaultWatcherManager()->getGui().setup(context->projectName);
    Bela_getDefaultWatcherManager()->setup(context->audioSampleRate);

    // Initialize Watchers for each sensor
    for (unsigned int i = 0; i < NUM_SENSORS; ++i) {
        gFaabWatchers.push_back(new Watcher<float>("gFaabSensor_" + std::to_string(i + 1)));
    }

    // Read sensor values into gSensorPlayback
    size_t minLength = 0;
    for (unsigned int i = 0; i < NUM_SENSORS; ++i) {
        std::vector<float> buffer = loadBufferData("data/gFaabSensor_" + std::to_string(i + 1) + ".bin", gBufferSize);
        gSensorPlayback[i] = buffer;

        if (i == 0) {
            minLength = buffer.size();
        } else if (buffer.size() < minLength) {
            minLength = buffer.size();
        }
    }

    // remove last samples to avoid big click
    gSensorPlaybackSize = minLength - 300;
    for (unsigned int i = 0; i < NUM_SENSORS; ++i) {
        gSensorPlayback[i].resize(gSensorPlaybackSize);
    }

    return true;
}

void render(BelaContext* context, void* userData) {
    for (unsigned int n = 0; n < context->audioFrames; n++) {
        uint64_t frames = context->audioFramesElapsed + n;
        Bela_getDefaultWatcherManager()->tick(frames);

        if (gAudioFramesPerAnalogFrame && !(n % gAudioFramesPerAnalogFrame)) {
            // Read sensor values and put them in the watcher
            // float out = 0;
            for (unsigned int i = 0; i < NUM_SENSORS; i++) {
                // Assign a single float value to the Watcher
                *gFaabWatchers[i] = gSensorPlayback[i][gPlaybackIndex];
                // out += gSensorPlayback[i][gPlaybackIndex];
            }

            // Loop playback index to the beginning if we exceed the data length
            if (gPlaybackIndex++ >= gSensorPlaybackSize) {
                gPlaybackIndex = 0;
            }

            float out = gSensorPlayback[4][gPlaybackIndex]; // loudest sensor
            audioWrite(context, n, 0, gain * out);
            audioWrite(context, n, 1, gain * out);
        }
    }
}

void cleanup(BelaContext* context, void* userData) {
}
