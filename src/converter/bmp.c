#include <stdio.h>

#include "../../include/stb_image.h"

#ifdef __cplusplus
extern "C" {
#endif

unsigned char* load_bmp_grayscale(const char* filename, int* width, int* height) {
    int channels = 0;
    unsigned char* img = stbi_load(filename, width, height, &channels, 1);
    if (!img) {
        fprintf(stderr, "Failed to load BMP image %s: %s\n", filename, stbi_failure_reason());
        return NULL;
    }
    return img;
}

#ifdef __cplusplus
}
#endif
