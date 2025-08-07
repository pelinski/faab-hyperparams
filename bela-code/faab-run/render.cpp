#include <Bela.h>
#include <Watcher.h>
#include <cmath>
#include <vector>

#define NUM_SENSORS 8

std::vector<Watcher<float>*> gFaabWatchers;
unsigned int gAudioFramesPerAnalogFrame;

bool setup(BelaContext* context, void* userData) {

    gAudioFramesPerAnalogFrame = context->audioFrames / context->analogFrames;

    Bela_getDefaultWatcherManager()->getGui().setup(context->projectName);
    Bela_getDefaultWatcherManager()->setup(context->audioSampleRate); // set sample rate in watcher

    // sensor watchers init
    for (unsigned int i = 0; i < NUM_SENSORS; ++i) {
        gFaabWatchers.push_back(new Watcher<float>("gFaabSensor_" + std::to_string(i + 1)));
    }

    return true;
}

void render(BelaContext* context, void* userData) {
    for (unsigned int n = 0; n < context->audioFrames; n++) {
        uint64_t frames = context->audioFramesElapsed + n;
        Bela_getDefaultWatcherManager()->tick(frames);

        if (gAudioFramesPerAnalogFrame && !(n % gAudioFramesPerAnalogFrame)) {

            // read sensor values and put them in the watcher
            for (unsigned int i = 0; i < NUM_SENSORS; i++) {
                *gFaabWatchers[i] = analogRead(context, n / gAudioFramesPerAnalogFrame, i);
            }

        }
    }
}

void cleanup(BelaContext* context, void* userData) {
}
