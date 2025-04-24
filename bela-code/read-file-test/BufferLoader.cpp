#include "BufferLoader.h"
#include <fstream>
#include <string>

std::string readNullTerminatedString(std::ifstream& file) {
    std::string result;
    char c;
    while (file.get(c)) {
        if (c == '\0')
            break;
        result += c;
    }
    return result;
}

std::vector<float> loadBufferData(const std::string& filename, size_t bufferSize) {
    std::vector<float> buffer;
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "Error: Could not open file '%s'\n", filename.c_str());
        return buffer;
    }

    // --- Parse header ---
    std::string projectName = readNullTerminatedString(file);
    std::string varName = readNullTerminatedString(file);
    std::string typeStr = readNullTerminatedString(file);

    uint32_t pid = 0, pid_id = 0;
    file.read(reinterpret_cast<char*>(&pid), sizeof(pid));
    file.read(reinterpret_cast<char*>(&pid_id), sizeof(pid_id));

    // Calculate header size and pad to 4-byte boundary
    size_t headerSize = projectName.length() + varName.length() + typeStr.length() + 3 + sizeof(pid) + sizeof(pid_id);
    if (headerSize % 4 != 0)
        file.ignore(4 - (headerSize % 4)); // skip padding

    // --- Parse buffers ---
    const size_t floatSize = sizeof(float);
    const size_t timestampSize = sizeof(uint64_t);
    const size_t totalBufferBytes = timestampSize + bufferSize * floatSize;

    while (true) {
        char rawBuffer[totalBufferBytes];
        file.read(rawBuffer, totalBufferBytes);
        if (file.gcount() < totalBufferBytes)
            break;

        float* dataStart = reinterpret_cast<float*>(rawBuffer + timestampSize);
        buffer.insert(buffer.end(), dataStart, dataStart + bufferSize);
    }

    return buffer;
}
