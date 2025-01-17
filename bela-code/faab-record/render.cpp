#include <Bela.h>
#include <Watcher.h>
#include <cmath>
#include <libraries/Scope/Scope.h>

#define NUM_SENSORS 8 // Number of sensors
#define NUM_OUTPUTS 4 // Number of output parameters

unsigned int gAudioFramesPerAnalogFrame; // Number of audio frames per analog
                                         // frame
float gInvSampleRate;                    // 1/sample rate
float gInvAudioFramesPerAnalogFrame;     // 1/audio frames per analog frame

// Vector of Watcher pointers
std::vector<Watcher<float>*> gFaabWatchers;

// // Osc
// float gFrequency[NUM_SENSORS] = {100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0, 12800.0};
// float gPhase[NUM_SENSORS] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
// float gAmplitude[NUM_SENSORS] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

Scope scope;

float gIn[NUM_SENSORS];

bool setup(BelaContext* context, void* userData) {
    Bela_getDefaultWatcherManager()->getGui().setup(context->projectName);
    Bela_getDefaultWatcherManager()->setup(context->audioSampleRate); // set sample rate in watcher
    gAudioFramesPerAnalogFrame = context->audioFrames / context->analogFrames;
    gInvAudioFramesPerAnalogFrame = 1.0f / gAudioFramesPerAnalogFrame;
    gInvSampleRate = 1.0f / context->audioSampleRate;

    // Initialize the Watcher pointers and add them to the vector
    for (unsigned int i = 0; i < NUM_SENSORS; ++i) {
        gFaabWatchers.push_back(new Watcher<float>("gFaabSensor_" + std::to_string(i + 1)));
    }

    // Initialize the scope
    scope.setup(8, context->analogSampleRate);

    return true;
}

void render(BelaContext* context, void* userData) {
    for (unsigned int n = 0; n < context->audioFrames; n++) {
        uint64_t frames = context->audioFramesElapsed + n;
        Bela_getDefaultWatcherManager()->tick(frames);

        if (gAudioFramesPerAnalogFrame && !(n % gAudioFramesPerAnalogFrame)) {
            // read analog inputs and update frequency and amplitude
            // Depending on the sampling rate of the analog inputs, this will
            // happen every audio frame (if it is 44100)
            // or every two audio frames (if it is 22050)
            for (unsigned int i = 0; i < NUM_SENSORS; i++) {
                gIn[i] = analogRead(context, n / gAudioFramesPerAnalogFrame, i);
                *gFaabWatchers[i] = gIn[i];
                // gAmplitude[i] = gFaabWatchers[i]->get();
            }

            scope.log(gIn[0], gIn[1], gIn[2], gIn[3], gIn[4], gIn[5], gIn[6], gIn[7]);
        }

        // // audio output
        // float out = 0.0;
        // for (unsigned int i = 0; i < NUM_SENSORS; i++) {
        //     out += gAmplitude[i] * sinf(gPhase[i]);
        // }
        // for (unsigned int channel = 0; channel < context->audioOutChannels; channel++) {
        //     audioWrite(context, n, channel, out);
        // }

        // // osc
        // for (unsigned int i = 0; i < NUM_SENSORS; i++) {
        //     // Update and wrap phase of sine tone
        //     gPhase[i] += 2.0f * (float)M_PI * gFrequency[i] * gInvSampleRate;
        //     if (gPhase[i] > M_PI)
        //         gPhase[i] -= 2.0f * (float)M_PI;
        // }
    }
}

void cleanup(BelaContext* context, void* userData) {
}
