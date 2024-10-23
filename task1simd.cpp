#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <CL/cl.hpp>
#include <iomanip>

// CPU implementation of vector addition
std::vector<float> addVectorsCPU(const std::vector<float>& vec1,
    const std::vector<float>& vec2) {
    std::vector<float> result(vec1.size());
    for (size_t i = 0; i < vec1.size(); i++) {
        result[i] = vec1[i] + vec2[i];
    }
    return result;
}

// GPU OpenCL implementation
class GPUVectorAddition {
private:
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;
    cl::Kernel kernel;

    const std::string kernelSource = R"(
        __kernel void addVectors(__global const float4* vec1, 
                                 __global const float4* vec2, 
                                 __global float4* result) {
            size_t i = get_global_id(0);
            result[i] = vec1[i] + vec2[i];
        }
    )";

public:
    GPUVectorAddition() {
        // Initialize OpenCL
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        platform = platforms[0];

        // Get GPU device
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        device = devices[0];

        // Create context and command queue
        context = cl::Context(device);
        queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

        // Build program
        cl::Program::Sources sources(1, std::make_pair(kernelSource.c_str(), kernelSource.length()));
        program = cl::Program(context, sources);
        program.build();
        kernel = cl::Kernel(program, "addVectors");

        // Print device info
        std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    }

    std::vector<float> addVectors(const std::vector<float>& vec1,
        const std::vector<float>& vec2,
        double& executionTime) {
        size_t vectorSize = vec1.size();
        size_t vectorSizeAligned = (vectorSize + 3) / 4 * 4; // Align to multiple of 4
        std::vector<float> result(vectorSizeAligned);

        // Create buffers
        cl::Buffer buffer1(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            vectorSizeAligned * sizeof(float), const_cast<float*>(vec1.data()));
        cl::Buffer buffer2(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            vectorSizeAligned * sizeof(float), const_cast<float*>(vec2.data()));
        cl::Buffer bufferResult(context, CL_MEM_WRITE_ONLY,
            vectorSizeAligned * sizeof(float));

        // Set kernel arguments
        kernel.setArg(0, buffer1);
        kernel.setArg(1, buffer2);
        kernel.setArg(2, bufferResult);

        // Execute kernel and measure time
        cl::Event event;
        queue.enqueueNDRangeKernel(kernel, cl::NullRange,
            cl::NDRange(vectorSizeAligned / 4), cl::NullRange,
            nullptr, &event);
        queue.finish();

        // Calculate execution time using profiling
        cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        cl_ulong end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        executionTime = (end - start) * 1e-6; // Convert nanoseconds to milliseconds

        // Read result
        queue.enqueueReadBuffer(bufferResult, CL_TRUE, 0,
            vectorSizeAligned * sizeof(float), result.data());

        // Trim the result to the original size
        result.resize(vectorSize);

        return result;
    }
};

// Utility function to verify results
bool verifyResults(const std::vector<float>& expected,
    const std::vector<float>& actual,
    float tolerance = 1e-6) {
    if (expected.size() != actual.size()) return false;
    for (size_t i = 0; i < expected.size(); i++) {
        if (std::abs(expected[i] - actual[i]) > tolerance) return false;
    }
    return true;
}

int main() {
    // Initialize vectors
    const size_t vectorSize = 1024 * 1024;
    std::vector<float> vector1(vectorSize), vector2(vectorSize);

    // Generate random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-10.0f, 10.0f);
    for (size_t i = 0; i < vectorSize; i++) {
        vector1[i] = dis(gen);
        vector2[i] = dis(gen);
    }

    // CPU Implementation
    auto cpuStart = std::chrono::high_resolution_clock::now();
    std::vector<float> resultCPU = addVectorsCPU(vector1, vector2);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    double cpuTime = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();

    // GPU Implementation
    GPUVectorAddition gpu;
    double gpuTime;
    std::vector<float> resultGPU = gpu.addVectors(vector1, vector2, gpuTime);

    // Verify results
    bool resultsMatch = verifyResults(resultCPU, resultGPU);

    // Print results
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\nVector addition performance comparison (" << vectorSize << " elements):\n";
    std::cout << "CPU time: " << cpuTime << " ms\n";
    std::cout << "GPU time: " << gpuTime << " ms\n";
    std::cout << "Results match: " << (resultsMatch ? "Yes" : "No") << std::endl;

    return 0;
}
