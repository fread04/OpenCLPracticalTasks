#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <CL/cl.hpp>
#include <iomanip>

// CPU implementation of array reduction (sum)
float sumArrayCPU(const std::vector<float>& array) {
    float sum = 0.0f;
    for (size_t i = 0; i < array.size(); i++) {
        sum += array[i];
    }
    return sum;
}

// GPU OpenCL implementation for reduction
class GPUReduction {
private:
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;
    cl::Kernel kernel;

    const std::string kernelSource = R"(
        __kernel void reduce(__global const float4* input, 
                             __global float* output, 
                             const unsigned int size) {
            int globalId = get_global_id(0);
            int localSize = get_local_size(0);
            int localId = get_local_id(0);
            __local float4 localBuffer[64]; // Local memory for reduction (256 / 4)

            // Load data into local memory using float4
            if (globalId * 4 < size) {
                localBuffer[localId] = input[globalId];
            } else {
                localBuffer[localId] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            // Perform reduction in local memory
            for (int stride = localSize / 2; stride > 0; stride >>= 1) {
                if (localId < stride) {
                    localBuffer[localId] += localBuffer[localId + stride];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            // Write the result of this work group to global memory
            if (localId == 0) {
                float4 sum = localBuffer[0];
                output[get_group_id(0)] = sum.x + sum.y + sum.z + sum.w;
            }
        }
    )";

public:
    GPUReduction() {
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
        kernel = cl::Kernel(program, "reduce");

        // Print device info
        std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    }

    float reduce(const std::vector<float>& input, double& executionTime) {
        size_t inputSize = input.size();
        size_t globalSize = (inputSize + 3) / 4; // Adjust for float4
        size_t localSize = 64; // Must be a power of 2 for the kernel (256 / 4)

        // Create buffers
        cl::Buffer bufferInput(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            inputSize * sizeof(float), const_cast<float*>(input.data()));
        size_t outputSize = (globalSize + localSize - 1) / localSize; // Number of work groups
        cl::Buffer bufferOutput(context, CL_MEM_WRITE_ONLY, outputSize * sizeof(float));

        // Set kernel arguments
        kernel.setArg(0, bufferInput);
        kernel.setArg(1, bufferOutput);
        kernel.setArg(2, (unsigned int)inputSize);

        // Execute kernel and measure time
        cl::Event event;
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(globalSize), cl::NDRange(localSize), nullptr, &event);
        queue.finish();

        // Calculate execution time using profiling
        cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        cl_ulong end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        executionTime = (end - start) * 1e-6; // Convert nanoseconds to milliseconds

        // Read the output from the GPU
        std::vector<float> partialSums(outputSize);
        queue.enqueueReadBuffer(bufferOutput, CL_TRUE, 0, outputSize * sizeof(float), partialSums.data());

        // Sum the results from each work group on the host
        float sum = sumArrayCPU(partialSums);

        return sum;
    }
};

int main() {
    // Initialize array
    const size_t arraySize = 1024 * 1024;
    std::vector<float> array(arraySize);

    // Generate random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (size_t i = 0; i < arraySize; i++) {
        array[i] = dis(gen);
    }

    // CPU Implementation
    auto cpuStart = std::chrono::high_resolution_clock::now();
    float resultCPU = sumArrayCPU(array);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    double cpuTime = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();

    // GPU Implementation
    GPUReduction gpu;
    double gpuTime;
    float resultGPU = gpu.reduce(array, gpuTime);

    // Verify results
    bool resultsMatch = std::abs(resultCPU - resultGPU) / std::abs(resultCPU) < 1e-4;

    // Print results
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\nArray reduction performance comparison (" << arraySize << " elements):\n";
    std::cout << "CPU result: " << resultCPU << " | Time: " << cpuTime << " ms\n";
    std::cout << "GPU result: " << resultGPU << " | Time: " << gpuTime << " ms\n";
    std::cout << "Results match: " << (resultsMatch ? "Yes" : "No") << std::endl;

    return 0;
}
