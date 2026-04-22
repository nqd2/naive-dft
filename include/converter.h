#ifndef CONVERTER_H
#define CONVERTER_H

#ifdef __cplusplus
extern "C" {
#endif

unsigned char* load_png_grayscale(const char* filename, int* width, int* height);
unsigned char* load_jpeg_grayscale(const char* filename, int* width, int* height);
unsigned char* load_bmp_grayscale(const char* filename, int* width, int* height);
void converter_free(unsigned char* pixels);

#ifdef __cplusplus
}
#endif

#endif
