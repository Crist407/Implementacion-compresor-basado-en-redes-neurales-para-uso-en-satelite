#ifndef IO_HELPERS_H
#define IO_HELPERS_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Carga una imagen RAW (BSQ, uint16) y la convierte a float planar (B x H x W).
 * @param filename Ruta al archivo .raw
 * @param B Bandas
 * @param H Altura
 * @param W Anchura
 * @return Puntero (malloc) al tensor de floats, o NULL si falla.
 */
float* load_image_bsq_u16_to_planar_f32(const char* filename, int B, int H, int W);

/**
 * @brief Variante extendida con validaciones y opciones.
 * @param filename Ruta al archivo .raw
 * @param B Bandas
 * @param H Altura
 * @param W Anchura
 * @param expect_little_endian 1: archivo little-endian; 0: big-endian; -1: no forzar comprobación (sin swap)
 * @param scale_to_unit 1 para normalizar a [0,1] dividiendo por 65535, 0 deja valores originales
 * @return Puntero (malloc) al tensor de floats o NULL si falla.
 */
float* load_image_bsq_u16_to_planar_f32_ex(const char* filename, int B, int H, int W, int expect_little_endian, int scale_to_unit);

/**
 * @brief Guarda un tensor planar (C x H x W) en un archivo .bin (float32).
 * @param filename Ruta del archivo de salida.
 * @param tensor Puntero a los datos.
 * @param C Canales
 * @param H Altura
 * @param W Anchura
 * @return 0 en éxito, -1 en error.
 */
int save_tensor_planar_f32(const char* filename, const float* tensor, int C, int H, int W);

#ifdef __cplusplus
}
#endif

#endif // IO_HELPERS_H