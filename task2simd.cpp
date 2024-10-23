#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <CL/cl.hpp>
#include <iomanip>

// Structure to represent a 2D matrix
struct Matrix {
    std::vector<float> data;
    size_t rows, cols;

    Matrix(size_t r, size_t c) : rows(r), cols(c), data(r* c) {}

    float& at(size_t i, size_t j) { return data[i * cols + j]; }
    const float& at(size_t i, size_t j) const { return data[i * cols + j]; }
};

// CPU implementation of matrix multiplication
Matrix multiplyMatricesCPU(const Matrix& A, const Matrix& B) {
    if (A.cols != B.rows) {
        throw std::runtime_error("Invalid matrix dimensions");
    }

    Matrix C(A.rows, B.cols);
    for (size_t i = 0; i < A.rows; i++) {
        for (size_t j = 0; j < B.cols; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < A.cols; k++) {
                sum += A.at(i, k) * B.at(k, j);
            }
            C.at(i, j) = sum;
        }
    }
    return C;
}

// GPU OpenCL implementation
class GPUMatrixMultiplication {
private:
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;
    cl::Kernel kernel;

    // OpenCL kernel with local memory optimization
    const std::string kernelSource = R"(
        __kernel void multiplyMatrices(
            __global const float* A,
            __global const float* B,
            __global float* C,
            const int M, const int N, const int K,
            __local float* localA,
            __local float* localB)
        {
            const int BLOCK_SIZE = 16;
            
            int row = get_group_id(0) * BLOCK_SIZE + get_local_id(0);
            int col = get_group_id(1) * BLOCK_SIZE + get_local_id(1);
            
            float sum = 0.0f;
            
            for (int i = 0; i < K; i += BLOCK_SIZE) {
                // Load data into local memory
                if (row < M && i + get_local_id(1) < K)
                    localA[get_local_id(0) * BLOCK_SIZE + get_local_id(1)] = 
                        A[row * K + i + get_local_id(1)];
                else
                    localA[get_local_id(0) * BLOCK_SIZE + get_local_id(1)] = 0.0f;
                
                if (col < N && i + get_local_id(0) < K)
                    localB[get_local_id(0) * BLOCK_SIZE + get_local_id(1)] = 
                        B[(i + get_local_id(0)) * N + col];
                else
                    localB[get_local_id(0) * BLOCK_SIZE + get_local_id(1)] = 0.0f;
                
                barrier(CLK_LOCAL_MEM_FENCE);
                
                // Compute partial sum
                for (int k = 0; k < BLOCK_SIZE; k++) {
                    sum += localA[get_local_id(0) * BLOCK_SIZE + k] * 
                           localB[k * BLOCK_SIZE + get_local_id(1)];
                }
                
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            
            if (row < M && col < N) {
                C[row * N + col] = sum;
            }
        }
    )";

public:
    GPUMatrixMultiplication() {
        // Initialize OpenCL
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        platform = platforms[0];

        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        device = devices[0];

        context = cl::Context(device);
        queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

        cl::Program::Sources sources(1, std::make_pair(kernelSource.c_str(), kernelSource.length()));
        program = cl::Program(context, sources);
        program.build();
        kernel = cl::Kernel(program, "multiplyMatrices");

        std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    }

    Matrix multiply(const Matrix& A, const Matrix& B, double& executionTime) {
        if (A.cols != B.rows) {
            throw std::runtime_error("Invalid matrix dimensions");
        }

        Matrix C(A.rows, B.cols);
        const int BLOCK_SIZE = 16;

        // Add padding to matrices if necessary
        size_t paddedM = ((A.rows + 3) / 4) * 4;
        size_t paddedK = ((A.cols + 3) / 4) * 4;
        size_t paddedN = ((B.cols + 3) / 4) * 4;

        // Create padded buffers
        std::vector<float> paddedA(paddedM * paddedK, 0.0f);
        std::vector<float> paddedB(paddedK * paddedN, 0.0f);

        // Copy original data to padded buffers
        for (size_t i = 0; i < A.rows; ++i) {
            for (size_t j = 0; j < A.cols; ++j) {
                paddedA[i * paddedK + j] = A.at(i, j);
            }
        }
        for (size_t i = 0; i < B.rows; ++i) {
            for (size_t j = 0; j < B.cols; ++j) {
                paddedB[i * paddedN + j] = B.at(i, j);
            }
        }

        // Create OpenCL buffers with padded data
        cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            paddedA.size() * sizeof(float), paddedA.data());
        cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            paddedB.size() * sizeof(float), paddedB.data());
        cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY,
            paddedM * paddedN * sizeof(float));

        // Set kernel arguments
        kernel.setArg(0, bufferA);
        kernel.setArg(1, bufferB);
        kernel.setArg(2, bufferC);
        kernel.setArg(3, static_cast<int>(paddedM));
        kernel.setArg(4, static_cast<int>(paddedN));
        kernel.setArg(5, static_cast<int>(paddedK));
        kernel.setArg(6, cl::Local(BLOCK_SIZE * BLOCK_SIZE * sizeof(float)));
        kernel.setArg(7, cl::Local(BLOCK_SIZE * BLOCK_SIZE * sizeof(float)));

        // Calculate grid size
        cl::NDRange globalSize(
            ((paddedM + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE,
            ((paddedN + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE
        );
        cl::NDRange localSize(BLOCK_SIZE, BLOCK_SIZE);

        // Execute kernel
        cl::Event event;
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize, nullptr, &event);
        queue.finish();

        // Calculate execution time
        cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        cl_ulong end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        executionTime = (end - start) * 1e-6; // Convert nanoseconds to milliseconds

        // After kernel execution, copy back only the non-padded part
        std::vector<float> resultData(paddedM * paddedN);
        queue.enqueueReadBuffer(bufferC, CL_TRUE, 0,
            resultData.size() * sizeof(float), resultData.data());

        // Copy back to the original size matrix
        for (size_t i = 0; i < A.rows; ++i) {
            for (size_t j = 0; j < B.cols; ++j) {
                C.at(i, j) = resultData[i * paddedN + j];
            }
        }

        return C;
    }
};

// Utility function to verify results
bool verifyResults(const Matrix& expected, const Matrix& actual, float tolerance = 1e-3) {
    if (expected.rows != actual.rows || expected.cols != actual.cols) {
        std::cout << "Matrix dimensions don't match!" << std::endl;
        return false;
    }

    for (size_t i = 0; i < expected.rows; i++) {
        for (size_t j = 0; j < expected.cols; j++) {
            if (std::abs(expected.at(i, j) - actual.at(i, j)) > tolerance) {
                std::cout << "Mismatch at (" << i << ", " << j << "): "
                          << "Expected " << expected.at(i, j)
                          << ", Got " << actual.at(i, j) << std::endl;
                return false;
            }
        }
    }
    return true;
}

int main() {
    // Initialize matrices
    const size_t M = 1024; // Matrix A dimensions: M x K
    const size_t K = 1024; // Matrix A dimensions: M x K, Matrix B dimensions: K x N
    const size_t N = 1024; // Matrix B dimensions: K x N

    Matrix A(M, K), B(K, N);

    // Generate random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < K; j++) {
            A.at(i, j) = dis(gen);
        }
    }
    for (size_t i = 0; i < K; i++) {
        for (size_t j = 0; j < N; j++) {
            B.at(i, j) = dis(gen);
        }
    }

    // CPU Implementation
    std::cout << "Starting CPU computation...\n";
    auto cpuStart = std::chrono::high_resolution_clock::now();
    Matrix resultCPU = multiplyMatricesCPU(A, B);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    double cpuTime = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();

    // GPU Implementation
    std::cout << "Starting GPU computation...\n";
    GPUMatrixMultiplication gpu;
    double gpuTime;
    Matrix resultGPU = gpu.multiply(A, B, gpuTime);

    // Verify results
    bool resultsMatch = verifyResults(resultCPU, resultGPU);

    // Print results
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\nMatrix multiplication performance comparison ("
        << M << "x" << K << " * " << K << "x" << N << "):\n";
    std::cout << "CPU time: " << cpuTime << " ms\n";
    std::cout << "GPU time: " << gpuTime << " ms\n";
    std::cout << "Speed-up: " << cpuTime / gpuTime << "x\n";
    std::cout << "Results match: " << (resultsMatch ? "Yes" : "No") << std::endl;

    if (!resultsMatch) {
        std::cout << "Results don't match. Checking first few elements:" << std::endl;
        for (int i = 0; i < 5 && i < resultCPU.rows; ++i) {
            for (int j = 0; j < 5 && j < resultCPU.cols; ++j) {
                std::cout << "CPU[" << i << "][" << j << "] = " << resultCPU.at(i, j)
                          << ", GPU[" << i << "][" << j << "] = " << resultGPU.at(i, j) << std::endl;
            }
        }
    }

    return 0;
}
