#define _POSIX_C_SOURCE 200112L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/stat.h>
#include "sorteny_model.h"
#include "io_helpers.h"
#ifdef _OPENMP
#include <omp.h>
#endif

// Parámetros alineados con la implementación de referencia.
#define EXPECTED_BANDS 8
#define EXPECTED_DTYPE_U16 2
#define MOD_LAMBDA_SCALE 0.05f
#define MIN_LAMBDA 0.0f
#define DEFAULT_MAX_LAMBDA 0.125f
#define NORM_CONST 65535.0f

typedef struct {
    uint16_t bands;
    uint16_t height;
    uint16_t width;
    uint16_t datatype;
    uint16_t num_filters;
} CompressedHeader;

static float* allocate_tensor(size_t C, size_t H, size_t W, const char* name) {
    size_t bytes = C * H * W * sizeof(float);
    void* mem = NULL;
    int rc = posix_memalign(&mem, 64, bytes);
    float* ptr = (rc == 0) ? (float*)mem : NULL;
    if (!ptr) {
        fprintf(stderr, "Error: fallo de reserva para tensor '%s' (%zu bytes)\n", name, bytes);
        return NULL;
    }
    return ptr;
}

static size_t max4(size_t a, size_t b, size_t c, size_t d) {
    size_t m = a;
    if (b > m) m = b;
    if (c > m) m = c;
    if (d > m) m = d;
    return m;
}

// Inversión simple por Gauss-Jordan para matriz cuadrada pequeña.
static int invert_matrix(const float* A, float* A_inv, int n) {
    double aug[8][16];
    if (n <= 0 || n > 8) return -1;

    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c) aug[r][c] = (double)A[r * n + c];
        for (int c = 0; c < n; ++c) aug[r][n + c] = (r == c) ? 1.0 : 0.0;
    }

    for (int col = 0; col < n; ++col) {
        int pivot = col;
        double max_abs = fabs(aug[pivot][col]);
        for (int r = col + 1; r < n; ++r) {
            double v = fabs(aug[r][col]);
            if (v > max_abs) {
                max_abs = v;
                pivot = r;
            }
        }
        if (max_abs < 1e-12) return -1;

        if (pivot != col) {
            for (int c = 0; c < 2 * n; ++c) {
                double tmp = aug[col][c];
                aug[col][c] = aug[pivot][c];
                aug[pivot][c] = tmp;
            }
        }

        double diag = aug[col][col];
        for (int c = 0; c < 2 * n; ++c) aug[col][c] /= diag;

        for (int r = 0; r < n; ++r) {
            if (r == col) continue;
            double f = aug[r][col];
            if (f == 0.0) continue;
            for (int c = 0; c < 2 * n; ++c) aug[r][c] -= f * aug[col][c];
        }
    }

    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c) {
            A_inv[r * n + c] = (float)aug[r][n + c];
        }
    }
    return 0;
}

/**
 * Si spectral_synthesis_kernel.bin no se cargó, derivarla numéricamente
 * invirtiendo la spectral_analysis_kernel.
 */
static int ensure_spectral_synthesis(SORTENY_Model* model) {
    if (model->spectral_syn.dense.kernel &&
        model->spectral_syn.dense.C_in == model->spectral_syn.dense.C_out &&
        model->spectral_syn.dense.C_in > 0) {
        return 0; // ya cargada
    }

    if (!model->spectral_an.dense.kernel ||
        model->spectral_an.dense.C_in != model->spectral_an.dense.C_out ||
        model->spectral_an.dense.C_in == 0 ||
        model->spectral_an.dense.C_in > 8) {
        return -1;
    }

    int n = (int)model->spectral_an.dense.C_in;
    float invA[64];
    if (invert_matrix(model->spectral_an.dense.kernel, invA, n) != 0) return -1;

    float* W_syn = (float*)malloc((size_t)n * n * sizeof(float));
    if (!W_syn) return -1;

    // W = inv(A): aplicado como sum_j in[j] * W[j,i] = (W^T · in)[i]
    // que iguala tf.linalg.matvec(tf.linalg.matrix_transpose(inv(A)), x)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            W_syn[i * n + j] = invA[i * n + j];
        }
    }

    model->spectral_syn.dense.kernel = W_syn;
    model->spectral_syn.dense.C_in = (size_t)n;
    model->spectral_syn.dense.C_out = (size_t)n;
    fprintf(stderr, "[WARN] spectral_synthesis_kernel.bin no encontrado. Usando inversa numérica.\n");
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc < 4 || argc > 5) {
        fprintf(stderr, "Uso: %s <input.bin> <output.raw> <weights_dir> [max_lambda]\n", argv[0]);
        fprintf(stderr, "Ej:   %s output/latent.bin output/recon.raw weights/pesos_decoder 0.125\n", argv[0]);
        return 1;
    }

    const char* input_bin = argv[1];
    const char* output_raw = argv[2];
    const char* weights_dir = argv[3];
    float max_lambda = (argc >= 5) ? (float)atof(argv[4]) : DEFAULT_MAX_LAMBDA;
    if (max_lambda <= 0.0f) max_lambda = DEFAULT_MAX_LAMBDA;

    // Modo paridad estricta
    const char* strict_env = getenv("STRICT_PARITY");
    int strict_parity = (strict_env && strict_env[0] == '1');
    const char* use_half_even_env = getenv("USE_HALF_EVEN");
    int use_half_even = 1; // por defecto half-to-even para máxima paridad con Python
    if (use_half_even_env) use_half_even = (use_half_even_env[0] == '1');
    if (strict_parity) use_half_even = 1;

    if (strict_parity) {
        printf("[STRICT] Parity mode enabled: deterministic + half-to-even.\n");
#ifdef _OPENMP
        omp_set_dynamic(0);
        omp_set_num_threads(1);
#endif
    }

    int ret = 1;
    FILE* f_in = NULL;
    uint8_t* q_map = NULL;
    int32_t* latents = NULL;
    SORTENY_Model* model = NULL;

    float* mod_hidden = NULL;
    float* modulator = NULL;
    float* band_latent = NULL;
    float* ds = NULL;
    float* buf0 = NULL;
    float* buf1 = NULL;
    float* spectral_domain = NULL;
    float* output_image = NULL;

    // Validar tamaño del archivo de entrada
    struct stat st;
    if (stat(input_bin, &st) != 0) {
        fprintf(stderr, "Error: no se pudo leer metadatos de '%s'\n", input_bin);
        goto cleanup;
    }

    f_in = fopen(input_bin, "rb");
    if (!f_in) {
        fprintf(stderr, "Error: no se pudo abrir '%s'\n", input_bin);
        goto cleanup;
    }

    // Leer cabecera
    CompressedHeader h = {0};
    if (fread(&h, sizeof(uint16_t), 5, f_in) != 5) {
        fprintf(stderr, "Error: cabecera incompleta en '%s'\n", input_bin);
        goto cleanup;
    }

    if (h.bands != EXPECTED_BANDS) {
        fprintf(stderr, "Error: bands=%u no soportado (esperado %d)\n", h.bands, EXPECTED_BANDS);
        goto cleanup;
    }
    if (h.datatype != EXPECTED_DTYPE_U16) {
        fprintf(stderr, "Error: datatype=%u no soportado (esperado %d)\n", h.datatype, EXPECTED_DTYPE_U16);
        goto cleanup;
    }
    if (h.height == 0 || h.width == 0 || h.num_filters == 0) {
        fprintf(stderr, "Error: cabecera inválida (dimensiones o filtros cero)\n");
        goto cleanup;
    }
    if ((h.height % 16) != 0 || (h.width % 16) != 0) {
        fprintf(stderr, "Error: dimensiones no divisibles por 16 (%ux%u)\n", h.height, h.width);
        goto cleanup;
    }

    size_t H = h.height;
    size_t W = h.width;
    size_t B = h.bands;
    size_t C_lat = h.num_filters;
    size_t H4 = H / 16;
    size_t W4 = W / 16;
    size_t q_map_size = H4 * W4;
    size_t latent_count = B * C_lat * H4 * W4;
    size_t expected_bytes = 5 * sizeof(uint16_t) + q_map_size * sizeof(uint8_t) + latent_count * sizeof(int32_t);

    if ((size_t)st.st_size != expected_bytes) {
        fprintf(stderr, "Error: tamaño de bitstream inconsistente. esperado=%zu, real=%lld\n",
                expected_bytes, (long long)st.st_size);
        goto cleanup;
    }

    q_map = (uint8_t*)malloc(q_map_size);
    latents = (int32_t*)malloc(latent_count * sizeof(int32_t));
    if (!q_map || !latents) {
        fprintf(stderr, "Error: memoria insuficiente al cargar bitstream\n");
        goto cleanup;
    }

    if (fread(q_map, sizeof(uint8_t), q_map_size, f_in) != q_map_size) {
        fprintf(stderr, "Error: lectura incompleta del Q-map\n");
        goto cleanup;
    }
    if (fread(latents, sizeof(int32_t), latent_count, f_in) != latent_count) {
        fprintf(stderr, "Error: lectura incompleta de latentes\n");
        goto cleanup;
    }
    fclose(f_in); f_in = NULL;

    // v1: mapa Q constante
    uint8_t q0 = q_map[0];
    for (size_t i = 1; i < q_map_size; ++i) {
        if (q_map[i] != q0) {
            fprintf(stderr, "Error: Q-map no constante (v1 no soporta quality map espacial).\n");
            goto cleanup;
        }
    }

    float lambda_quant = ((float)q0 / 255.0f) * (max_lambda - MIN_LAMBDA) + MIN_LAMBDA;

    printf("=== SORTENY Decompressor ===\n");
    printf("Bitstream: B=%zu H=%zu W=%zu C_lat=%zu Q=%u lambda_q=%.6f\n",
           B, H, W, C_lat, q0, lambda_quant);

    // Cargar modelo
    printf("Cargando modelo desde '%s'...\n", weights_dir);
    model = load_model_weights(weights_dir);
    if (!model) goto cleanup;

    // Derivar espectral inversa si no existe
    if (ensure_spectral_synthesis(model) != 0) {
        fprintf(stderr, "Error: faltan pesos de transformada espectral inversa.\n");
        goto cleanup;
    }

    // Validaciones de pesos
    if (!model->modulating_mod.dense_0.kernel || !model->modulating_mod.dense_0.bias ||
        !model->modulating_mod.dense_1.kernel || !model->modulating_mod.dense_1.bias) {
        fprintf(stderr, "Error: faltan pesos de modulación\n");
        goto cleanup;
    }
    if (!model->synthesis_syn.conv_0.kernel || !model->synthesis_syn.conv_0.bias ||
        !model->synthesis_syn.conv_1.kernel || !model->synthesis_syn.conv_1.bias ||
        !model->synthesis_syn.conv_2.kernel || !model->synthesis_syn.conv_2.bias ||
        !model->synthesis_syn.conv_3.kernel || !model->synthesis_syn.conv_3.bias ||
        !model->synthesis_syn.igdn_0.beta || !model->synthesis_syn.igdn_0.gamma ||
        !model->synthesis_syn.igdn_1.beta || !model->synthesis_syn.igdn_1.gamma ||
        !model->synthesis_syn.igdn_2.beta || !model->synthesis_syn.igdn_2.gamma) {
        fprintf(stderr, "Error: faltan pesos de síntesis/IGDN\n");
        goto cleanup;
    }

    // Verificar compatibilidad de canales
    if (model->modulating_mod.dense_1.C_out != B * C_lat) {
        fprintf(stderr, "Error: salida del modulador (%zu) no coincide con bandas*filtros (%zu)\n",
                model->modulating_mod.dense_1.C_out, B * C_lat);
        goto cleanup;
    }

    size_t c0_in = model->synthesis_syn.conv_0.C_in;
    size_t c0_out = model->synthesis_syn.conv_0.C_out;
    size_t c1_in = model->synthesis_syn.conv_1.C_in;
    size_t c1_out = model->synthesis_syn.conv_1.C_out;
    size_t c2_in = model->synthesis_syn.conv_2.C_in;
    size_t c2_out = model->synthesis_syn.conv_2.C_out;
    size_t c3_in = model->synthesis_syn.conv_3.C_in;

    if ((C_lat % 4) != 0 || (C_lat / 4) != c0_in) {
        fprintf(stderr, "Error: mismatch canales stage0\n");
        goto cleanup;
    }
    if ((c0_out % 4) != 0 || (c0_out / 4) != c1_in) {
        fprintf(stderr, "Error: mismatch canales stage1\n");
        goto cleanup;
    }
    if ((c1_out % 4) != 0 || (c1_out / 4) != c2_in) {
        fprintf(stderr, "Error: mismatch canales stage2\n");
        goto cleanup;
    }
    if ((c2_out % 4) != 0 || (c2_out / 4) != c3_in) {
        fprintf(stderr, "Error: mismatch canales stage3\n");
        goto cleanup;
    }

    // Modulación inversa
    mod_hidden = allocate_tensor(model->modulating_mod.dense_0.C_out, 1, 1, "mod_hidden");
    modulator = allocate_tensor(model->modulating_mod.dense_1.C_out, 1, 1, "modulator");
    if (!mod_hidden || !modulator) goto cleanup;

    {
        float input_lambda[1] = { lambda_quant / MOD_LAMBDA_SCALE };
        apply_dense(mod_hidden, input_lambda, &model->modulating_mod.dense_0);
        apply_relu(mod_hidden, (int)model->modulating_mod.dense_0.C_out);
        apply_dense(modulator, mod_hidden, &model->modulating_mod.dense_1);
        apply_relu(modulator, (int)model->modulating_mod.dense_1.C_out);
        printf("  M[0]=%.4f, M[100]=%.4f\n", modulator[0], modulator[100]);
    }

    size_t HW = H * W;
    size_t plane4 = H4 * W4;
    size_t H0 = H4, W0 = W4;
    size_t H1 = H0 * 2, W1 = W0 * 2;
    size_t H2 = H1 * 2, W2 = W1 * 2;
    size_t H3 = H2 * 2, W3 = W2 * 2;

    size_t max_ds_elems = max4(c0_in * H1 * W1, c1_in * H2 * W2, c2_in * H3 * W3, c3_in * H * W);
    size_t max_conv_elems = max4(c0_out * H1 * W1, c1_out * H2 * W2, c2_out * H3 * W3,
                                 model->synthesis_syn.conv_3.C_out * H * W);

    band_latent = allocate_tensor(C_lat, H4, W4, "band_latent");
    ds = allocate_tensor(1, 1, max_ds_elems, "depth_to_space_buffer");
    buf0 = allocate_tensor(1, 1, max_conv_elems, "conv_buffer_0");
    buf1 = allocate_tensor(1, 1, max_conv_elems, "conv_buffer_1");
    spectral_domain = allocate_tensor(B, H, W, "spectral_domain");
    output_image = allocate_tensor(B, H, W, "output_image");
    if (!band_latent || !ds || !buf0 || !buf1 || !spectral_domain || !output_image) goto cleanup;

    // Synthesis Transform (8 bands)
    printf("Ejecutando Synthesis Transform (%zu bandas)...\n", B);
    for (size_t b = 0; b < B; ++b) {
        printf("  Banda %zu/%zu...\n", b + 1, B);

        // Desmodulación por banda/canal
        for (size_t c = 0; c < C_lat; ++c) {
            float m = modulator[b * C_lat + c];
            if (fabsf(m) < 1e-8f) m = 1e-8f;
            size_t base = ((b * C_lat) + c) * plane4;
            size_t out_base = c * plane4;
            for (size_t p = 0; p < plane4; ++p) {
                band_latent[out_base + p] = (float)latents[base + p] / m;
            }
        }

        // g_s: DepthToSpace -> Conv(corr=False) -> IGDN (x3) + Conv final
        apply_depth_to_space(ds, band_latent, (int)C_lat, (int)H0, (int)W0, 2);
        apply_conv2d_corr_false(buf0, ds, &model->synthesis_syn.conv_0, (int)H1, (int)W1);
        apply_igdn(buf1, buf0, &model->synthesis_syn.igdn_0, (int)H1, (int)W1);

        apply_depth_to_space(ds, buf1, (int)c0_out, (int)H1, (int)W1, 2);
        apply_conv2d_corr_false(buf0, ds, &model->synthesis_syn.conv_1, (int)H2, (int)W2);
        apply_igdn(buf1, buf0, &model->synthesis_syn.igdn_1, (int)H2, (int)W2);

        apply_depth_to_space(ds, buf1, (int)c1_out, (int)H2, (int)W2, 2);
        apply_conv2d_corr_false(buf0, ds, &model->synthesis_syn.conv_2, (int)H3, (int)W3);
        apply_igdn(buf1, buf0, &model->synthesis_syn.igdn_2, (int)H3, (int)W3);

        apply_depth_to_space(ds, buf1, (int)c2_out, (int)H3, (int)W3, 2);
        apply_conv2d_corr_false(buf0, ds, &model->synthesis_syn.conv_3, (int)H, (int)W);

        // Desnormalizar (* 65535) y guardar en dominio espectral
        size_t boff = b * HW;
        for (size_t p = 0; p < HW; ++p) {
            spectral_domain[boff + p] = buf0[p] * NORM_CONST;
        }
    }

    free(latents); latents = NULL;
    free(band_latent); band_latent = NULL;
    printf("  Synthesis Transform completada.\n");

    // Transformada espectral inversa
    printf("Ejecutando Transformada Espectral Inversa...\n");
    apply_spectral_synthesis(output_image, spectral_domain, &model->spectral_syn, (int)H, (int)W);

    // Guardar como RAW BSQ uint16 (half-to-even para máxima paridad con Python)
    printf("Guardando '%s'...\n", output_raw);
    if (save_image_bsq_u16_from_planar_f32(output_raw, output_image, (int)B, (int)H, (int)W, use_half_even) != 0) {
        goto cleanup;
    }

    printf("Decodificación completada. RAW reconstruido: %s\n", output_raw);
    ret = 0;

cleanup:
    if (f_in) fclose(f_in);
    if (q_map) free(q_map);
    if (latents) free(latents);
    if (model) free_model_weights(model);

    if (mod_hidden) free(mod_hidden);
    if (modulator) free(modulator);
    if (band_latent) free(band_latent);
    if (ds) free(ds);
    if (buf0) free(buf0);
    if (buf1) free(buf1);
    if (spectral_domain) free(spectral_domain);
    if (output_image) free(output_image);
    return ret;
}
