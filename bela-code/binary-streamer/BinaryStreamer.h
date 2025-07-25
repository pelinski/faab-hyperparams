#pragma once

#include <Bela.h>
#include <vector>
#include <string>
#include <functional>
#include <fstream>
#include <cstdint>

/**
 * BinaryStreamer - A double-buffered binary file streaming library
 *
 * Allows playback of large binary files with custom headers by loading chunks
 * into two alternating buffers while one is being used and the other is being filled.
 * Designed for binary files with the format:
 * - Header: project_name\0var_name\0type_str\0pid(4bytes)pid_id(4bytes)[padding]
 * - Data blocks: timestamp(8bytes) + buffer_data(N*4bytes floats)
 */
class BinaryStreamer {
public:
    /**
     * Constructor
     * @param bufferSize Size of each buffer in float elements
     * @param taskPriority Priority for the auxiliary loading task (0-99, higher = more priority)
     */
    BinaryStreamer(size_t bufferSize = 22050, int taskPriority = 90);

    /**
     * Destructor - cleans up auxiliary task
     */
    ~BinaryStreamer();

    /**
     * Load a binary file for streaming
     * @param filename Path to the binary file
     * @return true if successful, false otherwise
     */
    bool loadFile(const std::string& filename);

    /**
     * Get the next sample
     * Call this once per audio frame
     * @return Next float sample
     */
    float getNextSample();

    /**
     * Check if the streamer is ready (file loaded successfully)
     * @return true if ready for playback
     */
    bool isReady() const {
        return m_ready;
    }

    /**
     * Get total number of samples in the file
     * @return Total samples in file
     */
    size_t getNumSamples() const {
        return m_totalSamples;
    }

    /**
     * Get current playback position in samples
     * @return Current position in file
     */
    size_t getCurrentPosition() const;

    /**
     * Set callback for buffer underrun warnings
     * @param callback Function to call when buffer can't load in time
     */
    void setUnderrunCallback(std::function<void()> callback) {
        m_underrunCallback = callback;
    }

    /**
     * Seek to a specific position in the file
     * @param samplePosition Position in samples to seek to
     * @return true if successful
     */
    bool seek(size_t samplePosition);

    /**
     * Reset playback to beginning of file
     */
    void reset();

    /**
     * Get header information
     */
    const std::string& getProjectName() const {
        return m_projectName;
    }
    const std::string& getVarName() const {
        return m_varName;
    }
    const std::string& getTypeStr() const {
        return m_typeStr;
    }
    uint32_t getPid() const {
        return m_pid;
    }
    uint32_t getPidId() const {
        return m_pidId;
    }

    // Public method for auxiliary task callback (must be public for C callback)
    void fillBuffer();

private:
    // Buffer management
    std::vector<float> m_sampleBuffers[2];
    size_t m_bufferSize;
    size_t m_readPtr;
    size_t m_bufferReadPtr; // Position in file (in samples)
    int m_activeBuffer;
    volatile int m_doneLoadingBuffer;

    // File info
    std::string m_filename;
    size_t m_totalSamples;
    size_t m_dataStartPos; // File position where data blocks start
    size_t m_blockSize;    // Size of each data block in bytes
    bool m_ready;

    // Header info
    std::string m_projectName;
    std::string m_varName;
    std::string m_typeStr;
    uint32_t m_pid;
    uint32_t m_pidId;

    // Auxiliary task for background loading
    AuxiliaryTask m_fillBufferTask;
    int m_taskPriority;

    // Callback for underrun warnings
    std::function<void()> m_underrunCallback;

    // Private methods
    bool parseHeader();
    bool initializeBuffers();
    bool loadBufferFromFile(std::vector<float>& buffer, size_t startSample);
    std::string readNullTerminatedString(std::ifstream& file);
    void cleanup();
};

// C-style callback function for Bela auxiliary task
void binaryStreamerFillBufferCallback(void* userData);