#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#include "converter.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define PI 3.14159265358979323846

__global__ void dft2d_kernel(const float* input, float* output_real, float* output_imag, int width, int height) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    if (u < width && v < height) {
        float sum_real = 0;
        float sum_imag = 0;

        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                float angle = 2.0f * (float)PI * ((float)u * x / (float)width + (float)v * y / (float)height);
                float val = input[y * width + x];
                sum_real += val * cos(angle);
                sum_imag -= val * sin(angle);
            }
        }
        output_real[v * width + u] = sum_real;
        output_imag[v * width + u] = sum_imag;
    }
}

static void write_pgm(const char* filename, const unsigned char* data, int width, int height) {
    FILE* f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Failed to open output file: %s\n", filename);
        return;
    }
    fprintf(f, "P5\n%d %d\n255\n", width, height);
    fwrite(data, 1, (size_t)width * (size_t)height, f);
    fclose(f);
}

static int ends_with_ignore_case(const char* text, const char* suffix);

static int write_graph_image(const char* filename, const unsigned char* data, int width, int height) {
    if (ends_with_ignore_case(filename, ".png")) {
        int ok = stbi_write_png(filename, width, height, 1, data, width);
        if (!ok) {
            fprintf(stderr, "Failed to write PNG output file: %s\n", filename);
        }
        return ok;
    }

    if (ends_with_ignore_case(filename, ".pgm") || ends_with_ignore_case(filename, ".pmg")) {
        write_pgm(filename, data, width, height);
        return 1;
    }

    fprintf(stderr, "Unsupported graph output format for %s. Use .png or .pgm.\n", filename);
    return 0;
}

static void build_dft_visual(
    const float* real,
    const float* imag,
    unsigned char* out,
    int width,
    int height
) {
    float max_val = 0.0f;
    int n = width * height;
    float* magnitude = (float*)malloc((size_t)n * sizeof(float));
    if (!magnitude) {
        fprintf(stderr, "Failed to allocate memory for DFT magnitude.\n");
        return;
    }

    for (int i = 0; i < n; i++) {
        float re = real[i];
        float im = imag[i];
        float mag = logf(1.0f + sqrtf(re * re + im * im));
        magnitude[i] = mag;
        if (mag > max_val) {
            max_val = mag;
        }
    }

    if (max_val <= 0.0f) {
        memset(out, 0, (size_t)n);
        free(magnitude);
        return;
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int shifted_x = (x + width / 2) % width;
            int shifted_y = (y + height / 2) % height;
            int src_idx = shifted_y * width + shifted_x;
            int dst_idx = y * width + x;
            out[dst_idx] = (unsigned char)(255.0f * (magnitude[src_idx] / max_val));
        }
    }

    free(magnitude);
}

static int ends_with_ignore_case(const char* text, const char* suffix) {
    size_t text_len = strlen(text);
    size_t suffix_len = strlen(suffix);
    if (suffix_len > text_len) {
        return 0;
    }
    const char* tail = text + (text_len - suffix_len);
    for (size_t i = 0; i < suffix_len; i++) {
        char a = tail[i];
        char b = suffix[i];
        if (a >= 'A' && a <= 'Z') {
            a = (char)(a - 'A' + 'a');
        }
        if (b >= 'A' && b <= 'Z') {
            b = (char)(b - 'A' + 'a');
        }
        if (a != b) {
            return 0;
        }
    }
    return 1;
}

static unsigned char* load_grayscale(const char* filename, int* width, int* height) {
    if (ends_with_ignore_case(filename, ".png")) {
        return load_png_grayscale(filename, width, height);
    }
    if (ends_with_ignore_case(filename, ".jpg") || ends_with_ignore_case(filename, ".jpeg")) {
        return load_jpeg_grayscale(filename, width, height);
    }
    if (ends_with_ignore_case(filename, ".bmp")) {
        return load_bmp_grayscale(filename, width, height);
    }
    fprintf(stderr, "Unsupported input format for %s. Use png/jpeg/bmp.\n", filename);
    return NULL;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input.(png|jpg|jpeg|bmp)> [input_output.pgm] [graph_output.(png|pgm)]\n", argv[0]);
        return 1;
    }

    const char* input_path = argv[1];
    const char* grayscale_path = (argc >= 3) ? argv[2] : "input.pgm";
    const char* graph_path = (argc >= 4) ? argv[3] : "graph.png";

    int width = 0;
    int height = 0;
    unsigned char* gray_pixels = load_grayscale(input_path, &width, &height);
    if (!gray_pixels) {
        return 1;
    }

    write_pgm(grayscale_path, gray_pixels, width, height);

    size_t count = (size_t)width * (size_t)height;
    size_t size = count * sizeof(float);

    float* h_input = (float*)malloc(size);
    float* h_out_real = (float*)malloc(size);
    float* h_out_imag = (float*)malloc(size);
    unsigned char* h_graph = (unsigned char*)malloc(count);
    if (!h_input || !h_out_real || !h_out_imag || !h_graph) {
        fprintf(stderr, "Host memory allocation failed.\n");
        converter_free(gray_pixels);
        free(h_input);
        free(h_out_real);
        free(h_out_imag);
        free(h_graph);
        return 1;
    }

    for (size_t i = 0; i < count; i++) {
        h_input[i] = (float)gray_pixels[i];
    }

    // Device memory
    float *d_input = NULL, *d_out_real = NULL, *d_out_imag = NULL;
    cudaError_t err = cudaMalloc(&d_input, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_input failed: %s\n", cudaGetErrorString(err));
        converter_free(gray_pixels);
        free(h_input);
        free(h_out_real);
        free(h_out_imag);
        free(h_graph);
        return 1;
    }
    err = cudaMalloc(&d_out_real, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_out_real failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        converter_free(gray_pixels);
        free(h_input);
        free(h_out_real);
        free(h_out_imag);
        free(h_graph);
        return 1;
    }
    err = cudaMalloc(&d_out_imag, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_out_imag failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_out_real);
        converter_free(gray_pixels);
        free(h_input);
        free(h_out_real);
        free(h_out_imag);
        free(h_graph);
        return 1;
    }

    err = cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_out_real);
        cudaFree(d_out_imag);
        converter_free(gray_pixels);
        free(h_input);
        free(h_out_real);
        free(h_out_imag);
        free(h_graph);
        return 1;
    }

    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

    dft2d_kernel<<<blocks, threads>>>(d_input, d_out_real, d_out_imag, width, height);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_out_real);
        cudaFree(d_out_imag);
        converter_free(gray_pixels);
        free(h_input);
        free(h_out_real);
        free(h_out_imag);
        free(h_graph);
        return 1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_out_real);
        cudaFree(d_out_imag);
        converter_free(gray_pixels);
        free(h_input);
        free(h_out_real);
        free(h_out_imag);
        free(h_graph);
        return 1;
    }

    err = cudaMemcpy(h_out_real, d_out_real, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy D2H real failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_out_real);
        cudaFree(d_out_imag);
        converter_free(gray_pixels);
        free(h_input);
        free(h_out_real);
        free(h_out_imag);
        free(h_graph);
        return 1;
    }
    err = cudaMemcpy(h_out_imag, d_out_imag, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy D2H imag failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_out_real);
        cudaFree(d_out_imag);
        converter_free(gray_pixels);
        free(h_input);
        free(h_out_real);
        free(h_out_imag);
        free(h_graph);
        return 1;
    }

    build_dft_visual(h_out_real, h_out_imag, h_graph, width, height);
    if (!write_graph_image(graph_path, h_graph, width, height)) {
        cudaFree(d_input);
        cudaFree(d_out_real);
        cudaFree(d_out_imag);
        converter_free(gray_pixels);
        free(h_input);
        free(h_out_real);
        free(h_out_imag);
        free(h_graph);
        return 1;
    }

    cudaFree(d_input);
    cudaFree(d_out_real);
    cudaFree(d_out_imag);
    converter_free(gray_pixels);
    free(h_input);
    free(h_out_real);
    free(h_out_imag);
    free(h_graph);
    return 0;
}