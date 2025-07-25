#include <Bela.h>
#include "BinaryStreamer.h"
#include <vector>
#include <string>

// Vector of streamer instances
std::vector<BinaryStreamer*> gStreamers;

// Configuration
const size_t NUM_FILES = 4; // Change this to your desired number
const size_t BUFFER_SIZE = 22050;
const int BASE_PRIORITY = 90;

// File names to load
std::vector<std::string> gFileNames = {"gFaabSensor_1.bin", "gFaabSensor_2.bin", "gFaabSensor_3.bin", "gFaabSensor_4.bin"};

bool setup(BelaContext* context, void* userData) {
    // Create N streamer instances
    for (size_t i = 0; i < NUM_FILES; i++) {
        // Slightly lower priority for each subsequent streamer
        int priority = BASE_PRIORITY - i;
        gStreamers.push_back(new BinaryStreamer(BUFFER_SIZE, priority));

        // Set up underrun callback with file index
        gStreamers[i]->setUnderrunCallback([i]() { rt_printf("Warning: File %zu buffer underrun!\n", i); });

        // Load the file
        if (i < gFileNames.size()) {
            if (!gStreamers[i]->loadFile(gFileNames[i])) {
                printf("Failed to load file %zu: %s\n", i, gFileNames[i].c_str());
                return false;
            }
            printf("Loaded file %zu: %s (%s)\n", i, gFileNames[i].c_str(), gStreamers[i]->getVarName().c_str());
        }
    }

    return true;
}

void render(BelaContext* context, void* userData) {
    for (unsigned int n = 0; n < context->audioFrames; n++) {
        if (n % 2) {
            // Mix all streams together
            float mixedSample = 0.0f;

            for (size_t i = 0; i < gStreamers.size(); i++) {
                if (gStreamers[i]->isReady()) {
                    mixedSample += gStreamers[i]->getNextSample();
                }
            }

            // Apply mixing gain to prevent clipping
            mixedSample /= gStreamers.size();

            // Output to all channels
            for (unsigned int channel = 0; channel < context->audioOutChannels; channel++) {
                audioWrite(context, n, channel, mixedSample);
            }
        }
    }
}

void cleanup(BelaContext* context, void* userData) {
    // Clean up all streamers
    for (auto* streamer : gStreamers) {
        delete streamer;
    }
    gStreamers.clear();
}