#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <CL/cl.hpp>
#include <iomanip>
#include <algorithm>
#include <cmath>

// CPU implementation using std::sort
std::vector<int> sortCPU(const std::vector<int>& data) {
    std::vector<int> result = data;
    std::sort(result.begin(), result.end());
    return result;
}

// Function to verify if a number is a power of two
bool isPowerOfTwo(size_t n) {
    return (n & (n - 1)) == 0 && n != 0;
}

// Function to verify the results of sorting
bool verifyResults(const std::vector<int>& cpuResult, const std::vector<int>& gpuResult) {
    return cpuResult == gpuResult;
}

class GPUBitonicSort {
private:
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;
    cl::Kernel kernel;

    const std::string kernelSource = R"(
    __kernel void bitonicSort(__global int* data, const uint stage, const uint passOfStage, const uint arraySize) {
        uint id = get_global_id(0);
        uint pairDistance = 1 << (stage - passOfStage);
        uint blockWidth = 2 * pairDistance;
        uint leftId = (id % pairDistance) + ((id / pairDistance) * blockWidth);
        uint rightId = leftId + pairDistance;

        if (rightId < arraySize) {
            bool ascending = ((leftId / (1 << stage)) % 2) == 0;
            int leftElement = data[leftId];
            int rightElement = data[rightId];

            if ((leftElement > rightElement) == ascending) {
                data[leftId] = rightElement;
                data[rightId] = leftElement;
            }
        }
    }
)";

public:
    GPUBitonicSort() {
        // Get the OpenCL platform
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        platform = platforms[0];

        // Get the GPU device
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        device = devices[0];

        context = cl::Context(device);
        queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

        // Compile the kernel
        cl::Program::Sources sources(1, std::make_pair(kernelSource.c_str(), kernelSource.length()));
        program = cl::Program(context, sources);

        // Error handling during build
        try {
            program.build();
        }
        catch (...) {
            std::cout << "Error during program build. Please check your OpenCL installation." << std::endl;
            std::cout << "Build log: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
            return; // Early return if build fails
        }

        kernel = cl::Kernel(program, "bitonicSort");
        std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    }

    std::vector<int> sort(const std::vector<int>& data, double& executionTime) {
        size_t dataSize = data.size();
        if (!isPowerOfTwo(dataSize)) {
            throw std::runtime_error("Input size must be a power of two for bitonic sort.");
        }

        std::vector<int> result = data;

        cl::Buffer buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            dataSize * sizeof(int), result.data());

        double totalTime = 0.0;

        for (unsigned int stage = 1; stage <= log2(dataSize); ++stage) {
            for (unsigned int passOfStage = 1; passOfStage <= stage; ++passOfStage) {
                kernel.setArg(0, buffer);
                kernel.setArg(1, stage);
                kernel.setArg(2, passOfStage);
                kernel.setArg(3, static_cast<cl_uint>(dataSize));

                cl::Event event;
                queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                    cl::NDRange(dataSize / 2), cl::NullRange,
                    nullptr, &event);
                queue.finish();

                cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
                cl_ulong end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
                totalTime += (end - start) * 1e-6;
            }
        }

        executionTime = totalTime;
        queue.enqueueReadBuffer(buffer, CL_TRUE, 0,
            dataSize * sizeof(int), result.data());

        return result;
    }

private:
    bool isSortedBitonic(const std::vector<int>& data, unsigned int stage) {
        size_t size = data.size();
        size_t segmentSize = 1 << stage;
        for (size_t i = 0; i < size; i += segmentSize) {
            bool ascending = ((i / segmentSize) % 2) == 0;
            for (size_t j = 1; j < segmentSize; ++j) {
                if (ascending && data[i + j - 1] > data[i + j]) return false;
                if (!ascending && data[i + j - 1] < data[i + j]) return false;
            }
        }
        return true;
    }
};

bool compareArrays(const std::vector<int>& a, const std::vector<int>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

// Function to verify if an array is sorted
bool isSorted(const std::vector<int>& data) {
    for (size_t i = 1; i < data.size(); i++) {
        if (data[i] < data[i - 1]) {
            return false; // Not sorted
        }
    }
    return true; // Sorted
}

int main() {
    const size_t arraySize = 1024 * 1024; // 1M elements

    // Creating and initializing the original array
    std::vector<int> originalData(arraySize);
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_int_distribution<int> dis(-1000000, 1000000);

    for (size_t i = 0; i < arraySize; i++) {
        originalData[i] = dis(gen);
    }

    // Print original data
    std::cout << "Original data (first 10 elements): ";
    for (size_t i = 0; i < std::min(size_t(10), originalData.size()); ++i) {
        std::cout << originalData[i] << " ";
    }
    std::cout << std::endl;

    // Create copies for CPU and GPU
    std::vector<int> cpuData = originalData;
    std::vector<int> gpuData = originalData;

    // CPU sorting
    auto cpuStart = std::chrono::high_resolution_clock::now();
    std::vector<int> resultCPU = sortCPU(cpuData);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    double cpuTime = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();

    // GPU sorting
    GPUBitonicSort gpu;
    double gpuTime;
    std::vector<int> resultGPU = gpu.sort(gpuData, gpuTime);

    // Verify results
    bool resultsMatch = verifyResults(resultCPU, resultGPU);

    // Output results
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\nSorting performance comparison (" << arraySize << " elements):\n";
    std::cout << "CPU time: " << cpuTime << " ms\n";
    std::cout << "GPU time: " << gpuTime << " ms\n";
    std::cout << "Results match: " << (resultsMatch ? "Yes" : "No") << std::endl;

    // Display first and last elements for verification
    const size_t elementsToShow = 10;
    std::cout << "\nFirst " << elementsToShow << " sorted elements:\n";
    std::cout << "CPU: ";
    for (size_t i = 0; i < elementsToShow; i++) {
        std::cout << resultCPU[i] << " ";
    }
    std::cout << "\nGPU: ";
    for (size_t i = 0; i < elementsToShow; i++) {
        std::cout << resultGPU[i] << " ";
    }
    
    std::cout << "\n\nLast " << elementsToShow << " sorted elements:\n";
    std::cout << "CPU: ";
    for (size_t i = arraySize - elementsToShow; i < arraySize; i++) {
        std::cout << resultCPU[i] << " ";
    }
    std::cout << "\nGPU: ";
    for (size_t i = arraySize - elementsToShow; i < arraySize; i++) {
        std::cout << resultGPU[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
