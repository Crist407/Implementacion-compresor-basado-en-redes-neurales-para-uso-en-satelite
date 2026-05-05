#define _POSIX_C_SOURCE 200112L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "sorteny_model.h"
#include "io_helpers.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#define DEBUG_DUMP_DIR "debug_dumps"

// Dimensiones del Modelo
#define BANDS 8
#define H_IN 512
#define W_IN 512

// Dimensiones AnalysisTransform (strides=2)
#define C0 128
#define H1 (H_IN / 2)   // 256
#define W1 (W_IN / 2)   // 256
#define C1 128
#define H2 (H1 / 2)     // 128
#define W2 (W1 / 2)     // 128
#define C2 128
#define H3 (H2 / 2)     // 64
#define W3 (W2 / 2)     // 64
#define C3_AN 384       // Canales de salida de Analysis (Latent)
#define H4 (H3 / 2)     // 32
#define W4 (W3 / 2)     // 32

// Dimensiones ModulatingTransform
#define MOD_IN 1        // 1 valor (lambda)
#define MOD_HIDDEN 192  // Capa oculta
#define MOD_OUT (BANDS * C3_AN) // 8 * 384 = 3072
// Escalado interno de la capa Lambda del ModulatingTransform: x / maxval
#define MOD_LAMBDA_SCALE 0.05f  // maxval del modelo Python
#define MIN_LAMBDA 0.0f  // Límite inferior de lambda según SORTENY.py

// Dimensiones Tensor Latente Final (Y)
#define C_Y_FLAT MOD_OUT // 3072
#define H_Y H4           // 32
#define W_Y W4           // 32

// TODAS ESTAS DIMENSIONES PODRÍAN CAMBIAR SI SE MODIFICA EL MODELO, TENERLO EN CUENTA.
// ESTAS DIMENSIONES LAS HEMOS SACADO DEL weights_index.json Y DEBEN COINCIDIR CON LOS PESOS CARGADOS.

// El normalizador de la primera capa Lambda de AnalysisTransform
#define NORM_CONST 65535.0f // (2**16) - 1

// Límite superior de lambda alineado con SORTENY.py
#define DEFAULT_MAX_LAMBDA 0.125f // 1/8


/**
 * @brief Reserva un tensor planar (C x H x W) en el heap.
 */
static float* allocate_tensor(size_t C, size_t H, size_t W, const char* name) {
    size_t bytes = C * H * W * sizeof(float);
    void* mem = NULL;
    int rc = posix_memalign(&mem, 64, bytes);
    float* ptr = (rc == 0) ? (float*)mem : NULL;
    if (!ptr) {
        fprintf(stderr, "Error: Fallo de reserva para el tensor: %s\n", name);
        return NULL;
    }
    return ptr;
}

static int env_is_enabled(const char* name) {
    const char* value = getenv(name);
    return value && value[0] == '1';
}

int main(int argc, char* argv[]) {
    if (argc < 4 || argc > 6) {
        fprintf(stderr, "Uso: %s <input_image.raw> <lambda> <output_Y_hat.bin> [model_dir] [max_lambda]\n", argv[0]);
        fprintf(stderr, "Ej:   %s data/T31TCG_...raw 0.1 output/latent.bin weights/encoder 0.125\n", argv[0]);
        return 1;
    }

    const char* input_file = argv[1];
    float lambda = (float)atof(argv[2]);
    const char* output_file = argv[3];
    const char* model_path = (argc >= 5) ? argv[4] : "weights/encoder";
    /*
     * max_lambda (opcional): rango superior usado para recortar lambda y
     * cuantizarla en el Q-map de 8 bits antes del ModulatingTransform.
     */
    float max_lambda = 0.0f;
    if (argc == 6) {
        max_lambda = (float)atof(argv[5]);
        if (max_lambda <= 0.0f) {
            fprintf(stderr, "[WARN] max_lambda <= 0. Ignorando escalado externo.\n");
            max_lambda = DEFAULT_MAX_LAMBDA;
        }
    } else {
        max_lambda = DEFAULT_MAX_LAMBDA;
    }
    // Recorte alineado con Python/C: lambda en [0, max_lambda]
    if (lambda < 0.0f) lambda = 0.0f;
    if (lambda > max_lambda) lambda = max_lambda;

    int ret = 1; // por defecto error

    // Punteros inicializados a NULL para cleanup seguro
    SORTENY_Model* model = NULL;
    float* in_image_tensor = NULL;
    float* spectral_out = NULL;
    float* scratch_a = NULL; // tamaño máximo: C0 x H1 x W1
    float* scratch_b = NULL; // tamaño máximo: C0 x H1 x W1
    float* band_normalized = NULL; // Buffer para normalización Lambda x/65535
    float* band_Y = NULL;
    int32_t* band_q_i32 = NULL;
    float* mod_hidden = NULL;
    float* modulator  = NULL;
    FILE* f_out = NULL;

    // Modo de paridad estricta: fuerza half-to-even y ejecución determinista
    const char* strict_env = getenv("STRICT_PARITY");
    int strict_parity = (strict_env && strict_env[0] == '1');
    if (strict_parity) {
        printf("[STRICT] Parity mode enabled: deterministic + half-to-even.\n");
#ifdef _OPENMP
        omp_set_dynamic(0);
        omp_set_num_threads(1);
#endif
    }

    // 1. Cargar Pesos
    printf("Cargando modelo desde '%s'...\n", model_path);
    model = load_model_weights(model_path);
    if (model == NULL) {
        goto cleanup;
    }

    // 2. Cargar Imagen de Entrada (RAW sin normalizar)
    // Forzamos little-endian (1) y NO escalamos aquí (0) para replicar Python
    in_image_tensor = load_image_bsq_u16_to_planar_f32_ex(input_file, BANDS, H_IN, W_IN, 1, 0); 
    if (in_image_tensor == NULL) {
        goto cleanup;
    }


    // 3. Reservar Memoria para Tensores Intermedios
    printf("Reservando memoria para tensores intermedios...\n");
    spectral_out = allocate_tensor(BANDS, H_IN, W_IN, "spectral_out");
    scratch_a = allocate_tensor(C0, H1, W1, "scratch_a");
    scratch_b = allocate_tensor(C0, H1, W1, "scratch_b");
    band_Y = allocate_tensor(C3_AN, H4, W4, "band_Y");
    mod_hidden = allocate_tensor(MOD_HIDDEN, 1, 1, "mod_hidden");
    modulator  = allocate_tensor(MOD_OUT, 1, 1, "modulator (M)");
    
    if (env_is_enabled("DUMP_STAGES")) {
        fprintf(stderr, "[WARN] DUMP_STAGES es legacy y no genera artefactos completos en el encoder actual; se ignora.\n");
    }

    if (!spectral_out || !scratch_a || !scratch_b || !band_Y || !mod_hidden || !modulator) {
        fprintf(stderr, "Error: fallo de reserva de memoria para tensores.\n");
        goto cleanup;
    }


    // 4. --- INICIO DEL PIPELINE DE INFERENCIA ---
    printf("Iniciando pipeline de inferencia...\n");
    
    // Etapa 1: Transformada Espectral (Todas las bandas a la vez)
    printf("  (1/5) Ejecutando Transformada Espectral...\n");
    apply_spectral_analysis(spectral_out, in_image_tensor, &model->spectral_an, H_IN, W_IN);
    
    
    printf("  (2/5) Init Analysis buffers...\n");
    free(in_image_tensor); in_image_tensor = NULL; // Liberamos la imagen original

    // Etapa 2: Modulating Transform (Calcular el factor de escala 'M') antes del bucle de bandas
    if (max_lambda > 0.0f) {
        printf("  (3/5) Ejecutando Modulating Transform (Lambda=%.6f, max_lambda=%.6f)\n", lambda, max_lambda);
    } else {
        printf("  (3/5) Ejecutando Modulating Transform (Lambda=%.6f, sin max_lambda)\n", lambda);
    }

    // Cuantización de lambda como en Python: Q = round((lambda - min_lambda)/(max_lambda - min_lambda)*255)
    uint8_t q_byte = (uint8_t)lrintf(((lambda - MIN_LAMBDA) / (max_lambda - MIN_LAMBDA)) * 255.0f);
    // Decuantización: lambda_quant = (Q/255)*(max_lambda - min_lambda) + min_lambda
    float lambda_quant = ((float)q_byte / 255.0f) * (max_lambda - MIN_LAMBDA) + MIN_LAMBDA;

    // El modulador tiene una capa Lambda interna que hace: x / maxval (donde maxval=0.05)
    // Esta división es NECESARIA para replicar el comportamiento de TensorFlow
    float input_lambda[1] = { lambda_quant / MOD_LAMBDA_SCALE };
    apply_dense(mod_hidden, input_lambda, &model->modulating_mod.dense_0);
    apply_relu(mod_hidden, MOD_HIDDEN);
    apply_dense(modulator, mod_hidden, &model->modulating_mod.dense_1);
    
    // Aplicar ReLU final a la salida (UNA SOLA VEZ)
    apply_relu(modulator, MOD_OUT);
    printf("        ...Modulador 'M' calculado. input_lambda=%.4f, M[0]=%.2f, M[100]=%.2f\n", 
           input_lambda[0], modulator[0], modulator[100]);
    
    // Preparar salida en streaming
    f_out = fopen(output_file, "wb");
    if (!f_out) { fprintf(stderr, "Error: no se pudo abrir '%s' para escritura.\n", output_file); goto cleanup; }
    // Buffer de salida grande para reducir llamadas al sistema en escrituras largas.
    setvbuf(f_out, NULL, _IOFBF, 1 << 20);

    // Cabecera estilo Python: 5 x uint16 (bands, height, width, datatype, num_filters)
    uint16_t header[5];
    header[0] = (uint16_t)BANDS;
    header[1] = (uint16_t)H_IN;
    header[2] = (uint16_t)W_IN;
    header[3] = (uint16_t)2;       // datatype uint16
    header[4] = (uint16_t)C3_AN;   // num_filters
    size_t hwritten = fwrite(header, sizeof(uint16_t), 5, f_out);
    if (hwritten != 5) { fprintf(stderr, "Error: escritura de cabecera incompleta.\n"); fclose(f_out); f_out=NULL; goto cleanup; }

    // Mapa Q (32x32) constante: lambda cuantizada a 8 bits
    uint8_t q_map[H4 * W4];
    for (size_t i = 0; i < H4 * W4; ++i) q_map[i] = q_byte;
    size_t qwritten = fwrite(q_map, sizeof(uint8_t), H4 * W4, f_out);
    if (qwritten != H4 * W4) { fprintf(stderr, "Error: escritura de Q incompleta.\n"); fclose(f_out); f_out=NULL; goto cleanup; }

    if (env_is_enabled("DEBUG_DUMP") || env_is_enabled("DUMP_SPECTRAL") ||
        env_is_enabled("DUMP_Y_PRE") || env_is_enabled("DUMP_M") ||
        env_is_enabled("DUMP_Y_FLOAT")) {
        fprintf(stderr, "[WARN] Los volcados DUMP_* del encoder son legacy y no se emiten en la ruta optimizada actual.\n");
    }

    // Etapa 3: Analysis Transform (Bucle sobre las 8 bandas) + modulación y cuantización por banda
    printf("  (4/5) Ejecutando Analysis Transform (Bucle de 8 bandas) y cuantización en streaming...\n"); 
    
    
    // Alocar buffer temporal para entrada normalizada
    band_normalized = (float*)malloc(H_IN * W_IN * sizeof(float));
    if (!band_normalized) {
        fprintf(stderr, "Error: No se pudo alocar memoria para band_normalized\n");
        goto cleanup;
    }


    // Configuración de redondeo (constante para todo el bucle)
    const char* use_half_even_env = getenv("USE_HALF_EVEN");
    int use_half_even = strict_parity ? 1 : (use_half_even_env && use_half_even_env[0] == '1');
    size_t plane_sz = (size_t)H_Y * W_Y;
    size_t band_latent_elems = (size_t)C3_AN * plane_sz;
    band_q_i32 = (int32_t*)malloc(band_latent_elems * sizeof(int32_t));
    if (!band_q_i32) {
        fprintf(stderr, "Error: No se pudo alocar memoria para buffer de latentes int32.\n");
        goto cleanup;
    }

    for (int b = 0; b < BANDS; b++) {
        const float* band_input_raw = spectral_out + (b * H_IN * W_IN);
        
        // Normalizar entrada (simular Lambda del modelo TF)
        for (int i = 0; i < H_IN * W_IN; i++) {
            band_normalized[i] = band_input_raw[i] / NORM_CONST;
        }
        

        
        // 1. Capa 0 (con entrada normalizada)
        apply_conv2d(scratch_a, band_normalized, &model->analysis_an.conv_0, H_IN, W_IN);


        // 2. GDN 0: scratch_a -> scratch_b
        apply_gdn(scratch_b, scratch_a, &model->analysis_an.gdn_0, H1, W1);
        

        // 3. Capa 1: scratch_b -> scratch_a (Reutilizamos A)
        apply_conv2d(scratch_a, scratch_b, &model->analysis_an.conv_1, H1, W1);
        
        // 4. GDN 1: scratch_a -> scratch_b (Reutilizamos B)
        apply_gdn(scratch_b, scratch_a, &model->analysis_an.gdn_1, H2, W2);
        
        // 5. Capa 2: scratch_b -> scratch_a (Reutilizamos A)
        apply_conv2d(scratch_a, scratch_b, &model->analysis_an.conv_2, H2, W2);
        
        // 6. GDN 2: scratch_a -> scratch_b (Reutilizamos B)
        apply_gdn(scratch_b, scratch_a, &model->analysis_an.gdn_2, H3, W3);
        
        // 7. Layer 3 (Output): scratch_b -> band_Y (sin GDN, sin bias)
        apply_conv2d(band_Y, scratch_b, &model->analysis_an.conv_3, H3, W3);
        
        // 8. Modulación + Cuantización por banda

        
        for (size_t c = 0; c < C3_AN; c++) {
            float M_val = modulator[b * C3_AN + c];
            size_t plane_off = c * plane_sz;
            for (size_t p = 0; p < plane_sz; p++) {
                float prod = band_Y[plane_off + p] * M_val;
                float qv_f = 0.0f;
                if (use_half_even) {
                    float _n = floorf(prod);
                    float _diff = prod - _n;
                    qv_f = (_diff > 0.5f) ? (_n + 1.0f) : (_diff < 0.5f ? _n : ((((int64_t)_n & 1LL) == 0LL) ? _n : (_n + 1.0f)));
                } else {
                    qv_f = roundf(prod);
                }
                band_q_i32[plane_off + p] = (int32_t)lrintf(qv_f);
            }

        }

        // Escribir toda la banda cuantizada en un solo bloque (orden band, channel, h, w).
        if (fwrite(band_q_i32, sizeof(int32_t), band_latent_elems, f_out) != band_latent_elems) {
            fprintf(stderr, "Error: escritura incompleta en salida (banda %d)\n", b);
            fclose(f_out); f_out=NULL; goto cleanup;
        }
    }

    printf("        ...Analysis Transform + cuantización completada.\n");

    // 5. --- FIN DEL PIPELINE DE INFERENCIA ---
    free(band_normalized); band_normalized = NULL;
    free(spectral_out); spectral_out = NULL;
    free(scratch_a); scratch_a = NULL;
    free(scratch_b); scratch_b = NULL;
    
    ret = 0;

cleanup:
    if (f_out) fclose(f_out);
    if (model) free_model_weights(model);
    
    if (in_image_tensor) free(in_image_tensor);
    if (spectral_out) free(spectral_out);
    if (band_Y) free(band_Y);
    if (mod_hidden) free(mod_hidden);
    if (modulator) free(modulator);
    if (band_normalized) free(band_normalized);
    if (band_q_i32) free(band_q_i32);

    return ret;
}
