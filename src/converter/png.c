#include <stdio.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../../include/stb_image.h"

#ifdef __cplusplus
extern "C" {
#endif

unsigned char* load_png_grayscale(const char* filename, int* width, int* height) {
    int channels = 0;
    unsigned char* img = stbi_load(filename, width, height, &channels, 1);
    if (!img) {
        fprintf(stderr, "Failed to load PNG image %s: %s\n", filename, stbi_failure_reason());
        return NULL;
    }
    return img;
}

void converter_free(unsigned char* pixels) {
    stbi_image_free(pixels);
}

#ifdef __cplusplus
}
#endif
