#ifndef SORTENY_MODEL_H
#define SORTENY_MODEL_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif


// --- ESTRUCTURAS DE DATOS PARA LOS PESOS --- 

// Estructura genérica para una capa Convolucional
typedef struct {
    float* kernel; // Forma: [kH, kW, C_in, C_out]
    float* bias;   // Forma: [C_out] o NULL si la capa no tiene bias
    
    // Dimensiones (para saber cómo leer los kernels)
    size_t kH, kW, C_in, C_out;

    // Paso de la convolución (strides_down del modelo Keras)
    int stride; // típicamente 1 o 2

    // Indicador de si existe bias
    int has_bias; // 1 si hay bias, 0 si no
    
} ConvLayer;

// Estructura para una capa GDN (también usada para IGDN)
typedef struct {
    float* beta;   // Forma: [C]
    float* gamma;  // Forma: [C, C]
    size_t C;
    float epsilon; // pequeña constante para evitar división por cero
} GDNLayer;

// Estructura para una capa Densa (Multiplicación de Matriz)
typedef struct {
    float* kernel; // Forma: [C_in, C_out]
    float* bias;   // Forma: [C_out]
    size_t C_in, C_out;
} DenseLayer;

// 1. Transformada Espectral (Análisis)
typedef struct {
    DenseLayer dense; // 8x8
} SpectralTransform;

// 2. Transformada de Análisis
typedef struct {
    // [Lambda], Conv, Conv, Conv, Conv
    ConvLayer conv_0;
    GDNLayer  gdn_0;
    ConvLayer conv_1;
    GDNLayer  gdn_1;
    ConvLayer conv_2;
    GDNLayer  gdn_2;
    ConvLayer conv_3; // Esta no tiene ni bias ni GDN
} AnalysisTransform;

// 3. Transformada de Modulación
typedef struct {
    // [Lambda], Dense, Dense
    DenseLayer dense_0;
    DenseLayer dense_1;
} ModulatingTransform;

// 4. Transformada de Síntesis (Inversa de Analysis)
//    Cada capa: depth_to_space(2) -> Conv2D(corr=False, stride=1) + IGDN
typedef struct {
    ConvLayer conv_0;   // 5x5, stride=1
    GDNLayer  igdn_0;   // IGDN (inverse GDN)
    ConvLayer conv_1;   // 5x5, stride=1
    GDNLayer  igdn_1;
    ConvLayer conv_2;   // 5x5, stride=1
    GDNLayer  igdn_2;
    ConvLayer conv_3;   // 5x5, stride=1, sin IGDN, con bias
} SynthesisTransform;

// 5. Transformada Espectral Inversa (Síntesis)
//    Guarda B^T donde B = inv(A), para uso directo con matvec
typedef struct {
    DenseLayer dense; // 8x8 (B^T)
} SpectralSynthesisTransform;


// --- ESTRUCTURA PRINCIPAL DEL MODELO ---
// Esta estructura contendrá TODOS los pesos cargados en RAM
typedef struct {
    SpectralTransform           spectral_an;
    AnalysisTransform           analysis_an;
    ModulatingTransform         modulating_mod;
    SynthesisTransform          synthesis_syn;    // Decodificador
    SpectralSynthesisTransform  spectral_syn;     // Espectral inversa
} SORTENY_Model;


// --- DECLARACIÓN DE FUNCIONES ---

/**
 * @brief Carga todos los pesos del modelo desde la carpeta 'pesos_bin/'.
 * Lee TSV para saber qué archivos cargar,
 * reserva memoria (malloc) para cada tensor en la estructura SORTENY_Model,
 * y lee los datos binarios de los archivos .bin.
 *
 * @param base_path Ruta a la carpeta 'pesos_bin/'.
 * @return Un puntero al modelo cargado, o NULL si falla.
 */
SORTENY_Model* load_model_weights(const char* base_path);

/**
 * @brief Libera toda la memoria (free) reservada por load_model_weights.
 */
void free_model_weights(SORTENY_Model* model);

/**
 * @brief Aplica la transformada espectral de análisis.
 * x' = A * x  (por píxel, cruza bandas)
 */
void apply_spectral_analysis(float* restrict out_tensor, const float* restrict in_tensor, 
                             const SpectralTransform* tf, int H, int W);

/**
 * @brief Aplica la transformada espectral de síntesis (inversa).
 * x_hat = B^T * x  (por píxel, cruza bandas)
 */
void apply_spectral_synthesis(float* restrict out_tensor, const float* restrict in_tensor,
                              const SpectralSynthesisTransform* tf, int H, int W);

/**
 * @brief Aplica una convolución 2D con correlación (corr=True).
 * Usada en la Analysis Transform. Kernel accedido en orden normal.
 */
void apply_conv2d(float* restrict out_tensor, const float* restrict in_tensor, 
                  const ConvLayer* layer, int H_in, int W_in);

/**
 * @brief Aplica una convolución 2D con corr=False (kernel rotado 180°).
 * Usada en la Synthesis Transform. Equivale a convolución (no correlación).
 */
void apply_conv2d_corr_false(float* restrict out_tensor, const float* restrict in_tensor,
                             const ConvLayer* layer, int H_in, int W_in);

/**
 * @brief Aplica la activación GDN (Generalized Divisive Normalization).
 * y = x / (beta + sum(gamma * |x|))
 */
void apply_gdn(float* restrict out_tensor, const float* restrict in_tensor,
               const GDNLayer* layer, int H, int W);

/**
 * @brief Aplica la activación IGDN (Inverse GDN).
 * y = x * (beta + sum(gamma * |x|))
 */
void apply_igdn(float* restrict out_tensor, const float* restrict in_tensor,
                const GDNLayer* layer, int H, int W);

/**
 * @brief Aplica depth_to_space: reordena canales a espacio.
 * (C, H, W) -> (C/block², H*block, W*block)
 */
void apply_depth_to_space(float* restrict out_tensor, const float* restrict in_tensor,
                          int C, int H, int W, int block_size);

/**
 * @brief Aplica una capa Densa.
 * y = x * W + b
 */
void apply_dense(float* restrict out_tensor, const float* restrict in_tensor, 
                 const DenseLayer* layer);

/**
 * @brief Aplica la activación ReLU.
 * y = max(0, x)
 */
void apply_relu(float* restrict tensor, int size);



#ifdef __cplusplus
}
#endif

#endif 