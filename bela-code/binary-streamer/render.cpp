#include <Bela.h>
#include "BinaryStreamer.h"

// Global streamer instance
BinaryStreamer gStreamer(22050, 90); // 22050 sample buffer, priority 90

bool setup(BelaContext* context, void* userData) {
    // Set up underrun warning callback
    gStreamer.setUnderrunCallback([]() { rt_printf("Warning: Binary buffer underrun! Consider increasing buffer size.\n"); });

    // Load your binary file
    if (!gStreamer.loadFile("gFaabSensor_1.bin")) {
        printf("Failed to load binary file!\n");
        return false;
    }

    printf("Loaded binary file:\n");
    printf("  Project: %s\n", gStreamer.getProjectName().c_str());
    printf("  Variable: %s\n", gStreamer.getVarName().c_str());
    printf("  Type: %s\n", gStreamer.getTypeStr().c_str());
    printf("  PID: %u, PID_ID: %u\n", gStreamer.getPid(), gStreamer.getPidId());
    printf("  Total samples: %zu\n", gStreamer.getNumSamples());

    return true;
}

void render(BelaContext* context, void* userData) {
    for (unsigned int n = 0; n < context->audioFrames; n++) {

        if (n % 2) {
            float sample = gStreamer.getNextSample();

            // Output to all channels
            for (unsigned int channel = 0; channel < context->audioOutChannels; channel++) {
                audioWrite(context, n, channel, sample);
            }
        }
    }
}

void cleanup(BelaContext* context, void* userData) {
    // BinaryStreamer destructor handles cleanup automatically
}