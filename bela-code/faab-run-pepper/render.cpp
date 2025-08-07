#include <Bela.h>
#include <RtThread.h>
#include <Watcher.h>
#include <cmath>
#include <vector>

#define NUM_SENSORS 8
#define NUM_OUTPUTS 4
#define MAX_EXPECTED_BUFFER_SIZE 1024

std::vector<Watcher<float>*> gFaabWatchers;
std::vector<std::vector<float>> circularBuffers(NUM_OUTPUTS); // +1 for the modelUpdateClock

size_t circularBufferSize = 30 * 1024;
size_t prefillSize = 3 * 1024;
uint32_t circularBufferWriteIndex[NUM_OUTPUTS] = {0};
uint32_t circularBufferReadIndex[NUM_OUTPUTS] = {0};

struct ReceivedBuffer {
    uint32_t bufferId;
    char bufferType[4];
    uint32_t bufferLen;
    uint32_t empty;
    std::vector<float> bufferData;
};
ReceivedBuffer receivedBuffer;
uint receivedBufferHeaderSize;
uint64_t totalReceivedCount;

// struct CallbackBufferCount {
//     uint32_t guiBufferId; // we're actually ignoring these
//     uint64_t count;
// };
// CallbackBufferCount callbackBufferCounts[NUM_OUTPUTS];

unsigned int gAudioFramesPerAnalogFrame;

// float risingThreshold = 0.021;               // Trigger to 1 above this threshold
// float fallingThreshold = 0.008;              // Trigger back to 0 below this threshold
// float schmittGate; // Schmitt trigger output

bool binaryDataCallback(const std::string& addr, const WSServerDetails* id, const unsigned char* data, size_t size, void* arg) {

    if (totalReceivedCount == 0) {
        RtThread::setThisThreadPriority(1);
    }

    totalReceivedCount++;

    std::memcpy(&receivedBuffer, data, receivedBufferHeaderSize);
    receivedBuffer.bufferData.resize(receivedBuffer.bufferLen);
    std::memcpy(receivedBuffer.bufferData.data(), data + receivedBufferHeaderSize,
                receivedBuffer.bufferLen * sizeof(float)); // data is a pointer to the beginning of the data

    // if ((data + receivedBufferHeaderSize) & 3) {
    //     fprintf(stderr, "data is not aligned\n");
    //     return true;
    // }

    int _id = receivedBuffer.bufferId;
    // if (_id < 0 || _id >= NUM_OUTPUTS) {
    //     fprintf(stderr, "Invalid buffer id: %d\n", _id);
    //     return true;
    // }

    // callbackBufferCounts[_id].count++;

    for (size_t i = 0; i < receivedBuffer.bufferLen; ++i) {
        circularBuffers[_id][circularBufferWriteIndex[_id]] = receivedBuffer.bufferData[i];
        // circularBuffers[_id][circularBufferWriteIndex[_id]] = ((float*)(data +
        // receivedBufferHeaderSize))[i];
        circularBufferWriteIndex[_id] = (circularBufferWriteIndex[_id] + 1) % circularBufferSize;
    }

    // printf("\ncallback buffer count:  %llu, total data size: %zu, bufferId: %d, "
    //        "bufferType: %s, bufferLen: %d \n",
    //        callbackBufferCounts[_id].count, size, receivedBuffer.bufferId, receivedBuffer.bufferType, receivedBuffer.bufferLen);

    return true;
}

bool setup(BelaContext* context, void* userData) {

    gAudioFramesPerAnalogFrame = context->audioFrames / context->analogFrames;

    Bela_getDefaultWatcherManager()->getGui().setup(context->projectName);
    Bela_getDefaultWatcherManager()->setup(context->audioSampleRate); // set sample rate in watcher

    // sensor watchers init
    for (unsigned int i = 0; i < NUM_SENSORS; ++i) {
        gFaabWatchers.push_back(new Watcher<float>("gFaabSensor_" + std::to_string(i + 1)));
    }

    // output buffers init
    for (int i = 0; i < NUM_OUTPUTS; ++i) {
        // callbackBufferCounts[i].guiBufferId =
        Bela_getDefaultWatcherManager()->getGui().setBuffer('f', MAX_EXPECTED_BUFFER_SIZE);
        // callbackBufferCounts[i].count = 0;
        circularBuffers[i].resize(circularBufferSize, 0.0f);
        // std::fill_n(std::back_inserter(circularBuffers[i]), prefillSize, 0.0f);
        // // prefill each circular buffer with prefillSize zeroes to give the write
        // pointer some time to catch up
        circularBufferWriteIndex[i] = prefillSize % circularBufferSize;
    }

    Bela_getDefaultWatcherManager()->getGui().setBinaryDataCallback(binaryDataCallback);

    receivedBufferHeaderSize = sizeof(receivedBuffer.bufferId) + sizeof(receivedBuffer.bufferType) + sizeof(receivedBuffer.bufferLen) + sizeof(receivedBuffer.empty);
    totalReceivedCount = 0;

    receivedBuffer.bufferData.reserve(MAX_EXPECTED_BUFFER_SIZE);

    // schmittGate = 0.0;

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

            // analog outputs
            for (unsigned int i = 0; i < NUM_OUTPUTS; i++) {
                float out = circularBuffers[i][circularBufferReadIndex[i]];
                analogWrite(context, n, i, out);
                if (totalReceivedCount > 0 && (circularBufferReadIndex[i] + 1) % circularBufferSize != circularBufferWriteIndex[i]) {
                    circularBufferReadIndex[i] = (circularBufferReadIndex[i] + 1) % circularBufferSize;
                }
                // else if (totalReceivedCount > 0) {
                //     rt_printf("Buffer %d full\n", i);
                // }
            }
            // // if audio input, output gate on ch 5 - this does not seem to work
            // float audioL = audioRead(context, n/gAudioFramesPerAnalogFrame, 0);
            // float filteredAudio = lpFilter.process(fabs(audioL));
            // if (filteredAudio > risingThreshold) {
            //     schmittGate = 1.0;
            // } else if (filteredAudio < fallingThreshold) {
            //     schmittGate = 0.0;
            // }
            // analogWrite(context, n, NUM_OUTPUTS, schmittGate);
        }
    }
}

void cleanup(BelaContext* context, void* userData) {
}
