#define _POSIX_C_SOURCE 200112L
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_BANDS 8
#define DEFAULT_HEIGHT 512
#define DEFAULT_WIDTH 512
#define DEFAULT_SIZE 256

typedef struct {
    const char* input_path;
    const char* output_path;
    int bands;
    int height;
    int width;
    int size;
} Options;

static void usage(const char* argv0) {
    fprintf(stderr,
        "Uso: %s --input scene.raw --output quicklook.ppm [opciones]\n\n"
        "Opciones:\n"
        "  --bands N      Bandas BSQ uint16 (default: 8)\n"
        "  --height N     Alto de imagen (default: 512)\n"
        "  --width N      Ancho de imagen (default: 512)\n"
        "  --size N       Tamano salida NxN PPM (default: 256)\n\n"
        "La salida es PPM P6 RGB usando Sentinel-2 8 bandas: RGB = B04,B03,B02.\n",
        argv0);
}

static int parse_int(const char* s, int* out) {
    char* end = NULL;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v < 1 || v > 1000000L) {
        return -1;
    }
    *out = (int)v;
    return 0;
}

static int parse_args(int argc, char** argv, Options* opt) {
    memset(opt, 0, sizeof(*opt));
    opt->bands = DEFAULT_BANDS;
    opt->height = DEFAULT_HEIGHT;
    opt->width = DEFAULT_WIDTH;
    opt->size = DEFAULT_SIZE;

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (strcmp(a, "--input") == 0 && i + 1 < argc) {
            opt->input_path = argv[++i];
        } else if (strcmp(a, "--output") == 0 && i + 1 < argc) {
            opt->output_path = argv[++i];
        } else if (strcmp(a, "--bands") == 0 && i + 1 < argc) {
            if (parse_int(argv[++i], &opt->bands) != 0) return -1;
        } else if (strcmp(a, "--height") == 0 && i + 1 < argc) {
            if (parse_int(argv[++i], &opt->height) != 0) return -1;
        } else if (strcmp(a, "--width") == 0 && i + 1 < argc) {
            if (parse_int(argv[++i], &opt->width) != 0) return -1;
        } else if (strcmp(a, "--size") == 0 && i + 1 < argc) {
            if (parse_int(argv[++i], &opt->size) != 0) return -1;
        } else if (strcmp(a, "-h") == 0 || strcmp(a, "--help") == 0) {
            usage(argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "Error: argumento desconocido o incompleto: %s\n", a);
            return -1;
        }
    }

    if (!opt->input_path || !opt->output_path || opt->bands < 3 || opt->size <= 0) {
        return -1;
    }
    return 0;
}

static uint16_t* load_raw_u16(const char* path, int bands, int height, int width) {
    size_t total = (size_t)bands * (size_t)height * (size_t)width;
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error: no se pudo abrir RAW '%s'\n", path);
        return NULL;
    }
    uint16_t* data = (uint16_t*)malloc(total * sizeof(uint16_t));
    if (!data) {
        fprintf(stderr, "Error: memoria insuficiente para RAW\n");
        fclose(f);
        return NULL;
    }
    size_t n = fread(data, sizeof(uint16_t), total, f);
    int extra = fgetc(f);
    fclose(f);
    if (n != total || extra != EOF) {
        fprintf(stderr, "Error: RAW '%s' debe tener exactamente %zu muestras uint16\n", path, total);
        free(data);
        return NULL;
    }
    return data;
}

static uint16_t percentile_u16(const uint16_t* band, size_t count, double pct) {
    unsigned int* hist = (unsigned int*)calloc(65536u, sizeof(unsigned int));
    if (!hist) return 0;
    for (size_t i = 0; i < count; ++i) {
        hist[band[i]]++;
    }
    size_t target = (size_t)((pct / 100.0) * (double)(count - 1u));
    size_t accum = 0;
    uint16_t value = 0;
    for (int v = 0; v < 65536; ++v) {
        accum += hist[v];
        if (accum > target) {
            value = (uint16_t)v;
            break;
        }
    }
    free(hist);
    return value;
}

static unsigned char stretch_u8(uint16_t v, uint16_t lo, uint16_t hi) {
    if (hi <= lo) {
        return 0;
    }
    if (v <= lo) return 0;
    if (v >= hi) return 255;
    unsigned int num = (unsigned int)(v - lo) * 255u;
    unsigned int den = (unsigned int)(hi - lo);
    return (unsigned char)((num + den / 2u) / den);
}

static int write_ppm(
    const char* path,
    const uint16_t* raw,
    int bands,
    int height,
    int width,
    int out_size
) {
    const size_t band_stride = (size_t)height * (size_t)width;
    (void)bands;
    const uint16_t* b02 = raw + 0 * band_stride;
    const uint16_t* b03 = raw + 1 * band_stride;
    const uint16_t* b04 = raw + 2 * band_stride;

    uint16_t r_lo = percentile_u16(b04, band_stride, 2.0);
    uint16_t r_hi = percentile_u16(b04, band_stride, 98.0);
    uint16_t g_lo = percentile_u16(b03, band_stride, 2.0);
    uint16_t g_hi = percentile_u16(b03, band_stride, 98.0);
    uint16_t b_lo = percentile_u16(b02, band_stride, 2.0);
    uint16_t b_hi = percentile_u16(b02, band_stride, 98.0);

    FILE* out = fopen(path, "wb");
    if (!out) {
        fprintf(stderr, "Error: no se pudo abrir salida '%s'\n", path);
        return -1;
    }
    if (fprintf(out, "P6\n%d %d\n255\n", out_size, out_size) < 0) {
        fclose(out);
        return -1;
    }

    for (int y = 0; y < out_size; ++y) {
        int sy = (int)(((long long)y * (long long)height) / (long long)out_size);
        if (sy >= height) sy = height - 1;
        for (int x = 0; x < out_size; ++x) {
            int sx = (int)(((long long)x * (long long)width) / (long long)out_size);
            if (sx >= width) sx = width - 1;
            size_t p = (size_t)sy * (size_t)width + (size_t)sx;
            unsigned char rgb[3];
            rgb[0] = stretch_u8(b04[p], r_lo, r_hi);
            rgb[1] = stretch_u8(b03[p], g_lo, g_hi);
            rgb[2] = stretch_u8(b02[p], b_lo, b_hi);
            if (fwrite(rgb, 1, 3, out) != 3) {
                fclose(out);
                return -1;
            }
        }
    }
    fclose(out);
    return 0;
}

int main(int argc, char** argv) {
    Options opt;
    if (parse_args(argc, argv, &opt) != 0) {
        usage(argv[0]);
        return 2;
    }

    uint16_t* raw = load_raw_u16(opt.input_path, opt.bands, opt.height, opt.width);
    if (!raw) return 1;
    if (write_ppm(opt.output_path, raw, opt.bands, opt.height, opt.width, opt.size) != 0) {
        fprintf(stderr, "Error: no se pudo escribir quicklook PPM\n");
        free(raw);
        return 1;
    }
    free(raw);
    printf("Quicklook generado: %s (%dx%d PPM P6)\n", opt.output_path, opt.size, opt.size);
    return 0;
}
