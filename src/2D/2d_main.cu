#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cufft.h>
#include <vector>
#include <algorithm>

#include "check_correct.hpp"
#include "helper_cuda/helper_cuda.h"

#define unit 8
#define rfft_size (unit * unit)

#define band_unit (8 )
#define shared_unit (unit * band_unit)

#define nwarp_in_block 1

// #include "rfft_2d/2d_rfft_1_naive.cu"
// #include "rfft_2d/2d_rfft_1_banksolve.cu"

#include "rfft_2d/2d_rfft_1_async.cu"

// #include "rfft_2d/2d_rfft_1_naive_2warp.cu"

// #include "rfft_2d/2d_rfft_2.cu"
// #include "rfft_2d/2d_rfft_2_banksolve.cu"

void printHelp()
{
    const char *helpMessage =
        "Program name: FlashFFTStencil-2D\n"
        "Usage: a.out [stencil-shape] [input_size] [time_step] \n"
        "Stencil-shape: Heat-2D / Box2D9P \n";
    printf("%s\n", helpMessage);
}


int main(int argc, char *argv[])
{

    if (argc != 4)
    {
        printHelp();
        return 1;
    }

    std::string kernel_shape = argv[1];
    const int INPUT_WIDTH = std::stoi(argv[2]);
    const int KERNEL_WIDTH = 3;
    const int time = std::stoi(argv[3]);

    const bool is_print_data = false;

    const int sub_input_width = unit - (KERNEL_WIDTH - 1); // TODO : 54,
    // const int sub_input_width = unit;
    const int OVERLAP_WIDTH = KERNEL_WIDTH - 1;

    if (INPUT_WIDTH % sub_input_width != 0)
    {
        std::cerr << "input_size % subinput_size != 0" << std::endl;
        std::cerr << "subinput_size = " << sub_input_width << std::endl;
        std::cerr << "input_size = " << INPUT_WIDTH << std::endl;
        return 0.0;
    }
    else
    {
        std::cout << "INFO: stencil kernel shape = " << kernel_shape << std::endl;
        std::cout << "INFO: input width = " << INPUT_WIDTH << std::endl;
        std::cout << "INFO: times step = " << time << std::endl;
    }

    const unsigned int block_num_x = (INPUT_WIDTH / sub_input_width) / 2 / nwarp_in_block;
    const unsigned int block_num_y = (INPUT_WIDTH / sub_input_width);

    const int ACTUAL_WIDTH = (INPUT_WIDTH / sub_input_width) * unit;

    const int gpu_input_size = (INPUT_WIDTH / sub_input_width) * (INPUT_WIDTH / sub_input_width) * rfft_size;
    const int cpu_input_size = (INPUT_WIDTH * INPUT_WIDTH);
    const int kernel_size = KERNEL_WIDTH * KERNEL_WIDTH;

    // malloc
    size_t mem_size_input_gpu = gpu_input_size * sizeof(double);
    size_t mem_size_output = cpu_input_size * sizeof(double);

    double *h_input_gpu = (double *)calloc(gpu_input_size, sizeof(double));

    double *h_input_cpu = (double *)calloc(cpu_input_size, sizeof(double));

    double *h_output = (double *)calloc(cpu_input_size, sizeof(double));

    std::vector<double> h_kernel(kernel_size);

    // 初始化输入数据
    for (int i = 0; i < INPUT_WIDTH; i++)
    {
        for (int j = 0; j < INPUT_WIDTH; j++)
        {
            if (is_print_data)
            {
                h_input_cpu[i * INPUT_WIDTH + j] = static_cast<double>(1);
            }
            else
            {
                h_input_cpu[i * INPUT_WIDTH + j] = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
            }

            int index_for_inputgpu = ((i / sub_input_width) * unit + i % sub_input_width) * ACTUAL_WIDTH + ((j / sub_input_width) * unit + j % sub_input_width);
            h_input_gpu[index_for_inputgpu] = h_input_cpu[i * INPUT_WIDTH + j];
        }
    }
    for (int i = 0; i < KERNEL_WIDTH; i++)
    {
        for (int j = 0; j < KERNEL_WIDTH; j++)
        {
            if (is_print_data)
            {
                h_kernel[i * KERNEL_WIDTH + j] = static_cast<double>((1));
            }
            else
            {
                h_kernel[i * KERNEL_WIDTH + j] = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
            }
        }
    }

    CreatePlan(h_kernel.data(), KERNEL_WIDTH, is_print_data);

    // malloc device memory
    double *d_input;
    checkCudaErrors(cudaMalloc((void **)&d_input, mem_size_input_gpu));
    double *d_output;
    checkCudaErrors(cudaMalloc((void **)&d_output, mem_size_output));
    checkCudaErrors(cudaMemcpy(d_input, h_input_gpu, mem_size_input_gpu, cudaMemcpyHostToDevice));

    // checkCudaErrors(cudaFuncSetCacheConfig(rfft_2d_8_nwarp<nwarp_in_block>, cudaFuncCachePreferL1)); // 最差
    // checkCudaErrors(cudaFuncSetCacheConfig(rfft_2d_8_nwarp<nwarp_in_block>, cudaFuncCachePreferShared));
    // checkCudaErrors(cudaFuncSetCacheConfig(rfft_2d_8_nwarp<nwarp_in_block>, cudaFuncCachePreferEqual));

    // checkCudaErrors(cudaFuncSetAttribute(rfft_2d_8_nwarp<nwarp_in_block>, cudaFuncAttributeMaxDynamicSharedMemorySize, (nwarp_in_block * 2 * shared_unit) * sizeof(double)));

	// Warm-up run used solely for performance evaluation; output correctness is not ensured.
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start, 0));
    rfft_2d_8_nwarp<nwarp_in_block><<<
        {block_num_x, block_num_y},
        // 1,
        nwarp_in_block * WARP_SIZE,
        (nwarp_in_block * 2 * shared_unit) * sizeof(double)
        // (nwarp_in_block * 2 * rfft_size) * sizeof(double)
        >>>(
        d_input,
        ACTUAL_WIDTH,
        INPUT_WIDTH,
        sub_input_width,
        OVERLAP_WIDTH,
        // fft_allnum - 1,
        d_output);
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    float single_run_time = 0;
    checkCudaErrors(cudaEventElapsedTime(&single_run_time, start, stop));
    const float target_warmup_time = 5000.0f;
    int estimated_iterations = 0;
    if(single_run_time > 0) {
        estimated_iterations = static_cast<int>(target_warmup_time / single_run_time);
    }
    int warmup_iterations = std::max(50, estimated_iterations);
    printf("Warmup iterations: %d (estimated for ~5s)\n", warmup_iterations);
    for(int warmup_iter = 0; warmup_iter < warmup_iterations; warmup_iter++){
		rfft_2d_8_nwarp<nwarp_in_block><<<
            {block_num_x, block_num_y},
            // 1,
            nwarp_in_block * WARP_SIZE,
            (nwarp_in_block * 2 * shared_unit) * sizeof(double)
            // (nwarp_in_block * 2 * rfft_size) * sizeof(double)
            >>>(
            d_input,
            ACTUAL_WIDTH,
            INPUT_WIDTH,
            sub_input_width,
            OVERLAP_WIDTH,
            // fft_allnum - 1,
            d_output);
        cudaDeviceSynchronize();
    }
    int time_i = 0;
    float elapsedTime = 0.0;
    checkCudaErrors(cudaEventRecord(start, 0));
    for(; time_i < time; time_i++)
    {
		rfft_2d_8_nwarp<nwarp_in_block><<<
            {block_num_x, block_num_y},
            // 1,
            nwarp_in_block * WARP_SIZE,
            (nwarp_in_block * 2 * shared_unit) * sizeof(double)
            // (nwarp_in_block * 2 * rfft_size) * sizeof(double)
            >>>(
            d_input,
            ACTUAL_WIDTH,
            INPUT_WIDTH,
            sub_input_width,
            OVERLAP_WIDTH,
            // fft_allnum - 1,
            d_output);
    }
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    std::cout << "Time = " << elapsedTime << "[ms]" << std::endl;
    double secs = elapsedTime / 1000.0;
    printf("GStencil/s = %f\n", (1.0 * INPUT_WIDTH * INPUT_WIDTH * time) / secs / 1e9);
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    cudaMemcpy(h_output, d_output, mem_size_output, cudaMemcpyDeviceToHost);

    // // 计算CPU结果
    // std::vector<std::vector<double>> new_kernel = rotateKernel180(h_kernel);
    // std::vector<std::vector<double>> new_input = convertTo2D(h_input_cpu, INPUT_WIDTH);
    // std::vector<double> ref_result = stencil2D(new_input, new_kernel);

    // if (is_print_data)
    // {
    //     std::ofstream outfile("output.txt");

    //     if (!outfile)
    //     {
    //         std::cerr << "无法打开文件!" << std::endl;
    //         return -1;
    //     }

    //     outfile << "输入数据：" << std::endl;
    //     for (int i = 0; i < ACTUAL_WIDTH; i++)
    //     {
    //         for (int j = 0; j < ACTUAL_WIDTH; j++)
    //         {
    //             outfile << h_input_gpu[i * ACTUAL_WIDTH + j] << ", ";
    //         }
    //         outfile << std::endl;
    //     }
    //     outfile << std::endl;

    //     outfile << "Kernel: " << std::endl;
    //     for (int i = 0; i < KERNEL_WIDTH; i++)
    //     {
    //         for (int j = 0; j < KERNEL_WIDTH; j++)
    //         {
    //             outfile << h_kernel[i * KERNEL_WIDTH + j] << "\t";
    //         }
    //         outfile << std::endl;
    //     }
    //     outfile << std::endl;

    //     outfile << "GPU计算结果：" << std::endl;
    //     for (int i = 0; i < INPUT_WIDTH; i++)
    //     {
    //         for (int j = 0; j < INPUT_WIDTH; j++)
    //         {
    //             outfile << h_output[i * INPUT_WIDTH + j] << "\t";
    //         }
    //         outfile << std::endl;
    //     }
    //     outfile << std::endl;

    //     outfile << "new input: " << std::endl;
    //     for (int i = 0; i < INPUT_WIDTH; i++)
    //     {
    //         for (int j = 0; j < INPUT_WIDTH; j++)
    //         {
    //             outfile << new_input[i][j] << "\t";
    //         }
    //         outfile << std::endl;
    //     }

    //     outfile << "new kernel: " << std::endl;
    //     for (int i = 0; i < KERNEL_WIDTH; i++)
    //     {
    //         for (int j = 0; j < KERNEL_WIDTH; j++)
    //         {
    //             outfile << new_kernel[i][j] << "\t";
    //         }
    //         outfile << std::endl;
    //     }

    //     outfile << "ref result: " << std::endl;
    //     for (int i = 0; i < INPUT_WIDTH; i++)
    //     {
    //         for (int j = 0; j < INPUT_WIDTH; j++)
    //         {
    //             outfile << ref_result[i * INPUT_WIDTH + j] << "\t";
    //         }
    //         outfile << std::endl;
    //     }

    //     outfile.close();
    // }

    // // 检查结果
    // if (areArraysEqual(h_output, ref_result.data(), INPUT_WIDTH * INPUT_WIDTH, 1e-6))
    // {
    //     std::cout << "Check correct!" << std::endl;
    // }
    // else
    // {
    //     std::cout << "Error: result wrong!" << std::endl;
    // }

    free(h_input_cpu);
    free(h_input_gpu);
    free(h_output);

    checkCudaErrors(cudaFree(d_input));
    checkCudaErrors(cudaFree(d_output));
}
