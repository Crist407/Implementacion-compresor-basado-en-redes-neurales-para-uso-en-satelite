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

// Estructura para una capa GDN
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

// 1. Transformada Espectral
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


// --- ESTRUCTURA PRINCIPAL DEL MODELO ---
// Esta estructura contendrá TODOS los pesos cargados en RAM
typedef struct {
    SpectralTransform      spectral_an;
    AnalysisTransform      analysis_an;
    ModulatingTransform    modulating_mod;
    
} SORTENY_Model;


// --- DECLARACIÓN DE FUNCIONES ---

/**
 * @brief Carga todos los pesos del modelo desde la carpeta 'pesos_bin/'.
 * * Lee TSV para saber qué archivos cargar,
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
 * @brief Aplica la transformada espectral (Etapa 1).
 * x' = A * x
 */
void apply_spectral_analysis(float* restrict out_tensor, const float* restrict in_tensor, 
                             const SpectralTransform* tf, int H, int W);

/**
 * @brief Aplica una convolución 2D (Etapa 2 y 4).
 * (Implementación con bucles for)
 */
void apply_conv2d(float* restrict out_tensor, const float* restrict in_tensor, 
                  const ConvLayer* layer, int H_in, int W_in);

/**
 * @brief Aplica la activación GDN (Etapa 2).
 */
void apply_gdn(float* restrict out_tensor, const float* restrict in_tensor,
               const GDNLayer* layer, int H, int W);

/**
 * @brief Aplica una capa Densa (Etapa 1 y 3).
 * y = x * W + b
 */
void apply_dense(float* restrict out_tensor, const float* restrict in_tensor, 
                 const DenseLayer* layer);

/**
 * @brief Aplica la activación ReLU (usada en Modulación e Hiper-Análisis).
 * y = max(0, x)
 */
void apply_relu(float* restrict tensor, int size);



#ifdef __cplusplus
}
#endif

#endif 