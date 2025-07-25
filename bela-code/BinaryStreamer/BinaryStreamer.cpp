#include "BinaryStreamer.h"
#include <algorithm>
#include <cstring>

// C-style callback function for Bela auxiliary task
void binaryStreamerFillBufferCallback(void* userData) {
    BinaryStreamer* streamer = static_cast<BinaryStreamer*>(userData);
    if (streamer) {
        streamer->fillBuffer();
    }
}

BinaryStreamer::BinaryStreamer(size_t bufferSize, int taskPriority)
    : m_bufferSize(bufferSize), m_readPtr(bufferSize) // Start at buffer end to trigger initial load
      ,
      m_bufferReadPtr(0), m_activeBuffer(0), m_doneLoadingBuffer(1), m_totalSamples(0), m_dataStartPos(0), m_blockSize(0), m_ready(false), m_pid(0), m_pidId(0), m_fillBufferTask(nullptr),
      m_taskPriority(taskPriority) {
}

BinaryStreamer::~BinaryStreamer() {
    cleanup();
}

std::string BinaryStreamer::readNullTerminatedString(std::ifstream& file) {
    std::string result;
    char c;
    while (file.get(c)) {
        if (c == '\0')
            break;
        result += c;
    }
    return result;
}
bool BinaryStreamer::parseHeader() {
    std::ifstream file(m_filename, std::ios::binary);
    if (!file.is_open()) {
        rt_printf("BinaryStreamer: Could not open file %s\n", m_filename.c_str());
        return false;
    }

    // Parse header components
    m_projectName = readNullTerminatedString(file);
    m_varName = readNullTerminatedString(file);
    m_typeStr = readNullTerminatedString(file);

    file.read(reinterpret_cast<char*>(&m_pid), sizeof(m_pid));
    file.read(reinterpret_cast<char*>(&m_pidId), sizeof(m_pidId));

    if (!file.good()) {
        rt_printf("BinaryStreamer: Error reading header\n");
        return false;
    }

    // Calculate header size and handle padding
    size_t headerSize = m_projectName.length() + m_varName.length() + m_typeStr.length() + 3 + sizeof(m_pid) + sizeof(m_pidId);
    if (headerSize % 4 != 0) {
        file.ignore(4 - (headerSize % 4)); // skip padding
        headerSize += 4 - (headerSize % 4);
    }

    m_dataStartPos = headerSize;

    // Calculate block structure - FIXED for 1024 samples per block
    const size_t SAMPLES_PER_BLOCK = 1024; // Actual samples per block in file
    const size_t timestampSize = sizeof(uint64_t);
    const size_t floatSize = sizeof(float);

    // This is the actual block size in the file
    m_blockSize = timestampSize + SAMPLES_PER_BLOCK * floatSize;

    // Calculate total samples by examining file size
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    size_t dataSize = fileSize - m_dataStartPos;
    size_t numBlocks = dataSize / m_blockSize;
    m_totalSamples = numBlocks * SAMPLES_PER_BLOCK; // Use actual samples per block

    rt_printf("BinaryStreamer: Loaded %s - %zu samples in %zu blocks\n", m_filename.c_str(), m_totalSamples, numBlocks);
    rt_printf("  Each block: %zu samples + 8-byte timestamp\n", SAMPLES_PER_BLOCK);

    return m_totalSamples > 0;
}

bool BinaryStreamer::loadFile(const std::string& filename) {
    cleanup(); // Clean up any existing file

    m_filename = filename;

    if (!parseHeader()) {
        return false;
    }

    if (m_totalSamples <= m_bufferSize) {
        rt_printf("BinaryStreamer: File too short for streaming (needs > %zu samples)\n", m_bufferSize);
        return false;
    }

    // Create auxiliary task for background loading, passing 'this' as userData
    m_fillBufferTask = Bela_createAuxiliaryTask(binaryStreamerFillBufferCallback, m_taskPriority, "binary-streamer", this);
    if (!m_fillBufferTask) {
        rt_printf("BinaryStreamer: Failed to create auxiliary task\n");
        return false;
    }

    if (!initializeBuffers()) {
        cleanup();
        return false;
    }

    m_ready = true;
    return true;
}

bool BinaryStreamer::initializeBuffers() {
    // Initialize both buffers
    m_sampleBuffers[0].resize(m_bufferSize, 0.0f);
    m_sampleBuffers[1].resize(m_bufferSize, 0.0f);

    // Load initial buffer
    if (!loadBufferFromFile(m_sampleBuffers[0], 0)) {
        rt_printf("BinaryStreamer: Failed to load initial buffer\n");
        return false;
    }

    // Reset state
    m_readPtr = m_bufferSize; // Will trigger load on first call
    m_bufferReadPtr = 0;
    m_activeBuffer = 0;
    m_doneLoadingBuffer = 1;

    return true;
}

bool BinaryStreamer::loadBufferFromFile(std::vector<float>& buffer, size_t startSample) {
    std::ifstream file(m_filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    const size_t SAMPLES_PER_BLOCK = 1024; // Actual block size in file
    const size_t timestampSize = sizeof(uint64_t);
    const size_t floatSize = sizeof(float);
    const size_t actualBlockSize = timestampSize + SAMPLES_PER_BLOCK * floatSize;

    size_t samplesLoaded = 0;
    size_t currentSample = startSample;

    while (samplesLoaded < m_bufferSize && file.good()) {
        // Calculate which block and position within block
        size_t blockIndex = currentSample / SAMPLES_PER_BLOCK;
        size_t sampleInBlock = currentSample % SAMPLES_PER_BLOCK;

        // Seek to the start of this block
        size_t blockPos = m_dataStartPos + blockIndex * actualBlockSize;
        file.seekg(blockPos);

        // Skip timestamp
        file.seekg(timestampSize, std::ios::cur);

        // Skip to the sample we need within this block
        if (sampleInBlock > 0) {
            file.seekg(sampleInBlock * floatSize, std::ios::cur);
        }

        // Read samples from this block
        size_t samplesRemainingInBlock = SAMPLES_PER_BLOCK - sampleInBlock;
        size_t samplesToRead = std::min(samplesRemainingInBlock, m_bufferSize - samplesLoaded);

        file.read(reinterpret_cast<char*>(&buffer[samplesLoaded]), samplesToRead * floatSize);

        if (!file.good())
            break;

        samplesLoaded += samplesToRead;
        currentSample += samplesToRead;
    }

    // Zero-pad if needed
    if (samplesLoaded < m_bufferSize) {
        std::fill(buffer.begin() + samplesLoaded, buffer.end(), 0.0f);
    }

    return true;
}

float BinaryStreamer::getNextSample() {
    if (!m_ready)
        return 0.0f;

    // Check if we need to switch buffers
    if (++m_readPtr >= m_bufferSize) {
        if (!m_doneLoadingBuffer) {
            // UNDERRUN: Buffer not ready, don't switch!
            if (m_underrunCallback) {
                m_underrunCallback();
            }
            m_readPtr--; // Stay on last sample
            return m_sampleBuffers[m_activeBuffer][m_bufferSize - 1];
        }

        // Buffer is ready, proceed with switch
        m_doneLoadingBuffer = 0;
        m_readPtr = 0;
        m_activeBuffer = !m_activeBuffer;

        // Schedule next buffer fill
        Bela_scheduleAuxiliaryTask(m_fillBufferTask);
    }

    return m_sampleBuffers[m_activeBuffer][m_readPtr];
}

void BinaryStreamer::fillBuffer() {
    if (!m_ready)
        return;

    // Increment buffer read pointer
    m_bufferReadPtr += m_bufferSize;

    // Loop back to beginning if we've reached the end
    if (m_bufferReadPtr >= m_totalSamples) {
        m_bufferReadPtr = 0;
    }

    // Load the inactive buffer
    loadBufferFromFile(m_sampleBuffers[!m_activeBuffer], m_bufferReadPtr);

    m_doneLoadingBuffer = 1;
}

size_t BinaryStreamer::getCurrentPosition() const {
    if (!m_ready)
        return 0;
    return m_bufferReadPtr + m_readPtr;
}

bool BinaryStreamer::seek(size_t samplePosition) {
    if (!m_ready)
        return false;

    // Clamp to valid range
    samplePosition = std::min(samplePosition, m_totalSamples - 1);

    // Align to buffer boundaries for simplicity
    m_bufferReadPtr = (samplePosition / m_bufferSize) * m_bufferSize;
    m_readPtr = samplePosition % m_bufferSize;

    // Force a buffer reload
    m_doneLoadingBuffer = 0;
    if (m_fillBufferTask) {
        Bela_scheduleAuxiliaryTask(m_fillBufferTask);
    }

    return true;
}

void BinaryStreamer::reset() {
    if (!m_ready)
        return;

    m_bufferReadPtr = 0;
    m_readPtr = m_bufferSize; // Will trigger reload
    m_activeBuffer = 0;
    m_doneLoadingBuffer = 1;
}

void BinaryStreamer::cleanup() {
    if (m_fillBufferTask) {
        // Note: Bela doesn't provide a way to explicitly destroy auxiliary tasks
        // They are cleaned up when the context ends
        m_fillBufferTask = nullptr;
    }

    m_sampleBuffers[0].clear();
    m_sampleBuffers[1].clear();
    m_ready = false;
}