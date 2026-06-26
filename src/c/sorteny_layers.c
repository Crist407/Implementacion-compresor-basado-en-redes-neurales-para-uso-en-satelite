#include "sorteny_model.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#if defined(USE_NEON)
#include <arm_neon.h>
#endif

// Implementación de las capas basicas: ReLU, Dense, SpectralTransform, Conv2D, GDN

void apply_relu(float* restrict tensor, int size) {
    for (int i = 0; i < size; i++) {
        if (tensor[i] < 0.0f) tensor[i] = 0.0f;
    }
}

void apply_dense(float* restrict out_tensor, const float* restrict in_tensor, 
                 const DenseLayer* layer) {
    size_t C_in = layer->C_in;
    size_t C_out = layer->C_out;

    // Inicializar con bias
    for (size_t c = 0; c < C_out; c++) out_tensor[c] = (layer->bias) ? layer->bias[c] : 0.0f;

    // Multiplicación de matrices estándar
    for (size_t i = 0; i < C_in; i++) {
        float x = in_tensor[i];
        const float* rowW = layer->kernel + i * C_out;
        for (size_t j = 0; j < C_out; j++) {
            out_tensor[j] += x * rowW[j];
        }
    }
}

void apply_spectral_analysis(float* restrict out_tensor, const float* restrict in_tensor, 
                             const SpectralTransform* tf, int H, int W) {
    // Sin cambios (suficientemente rápido)
    size_t B = tf->dense.C_in;
    size_t pixel_count = (size_t)H * (size_t)W;
#ifdef _OPENMP
    #pragma omp parallel for if (pixel_count >= 1024)
#endif
    for (size_t p = 0; p < pixel_count; p++) {
        for (size_t b_out = 0; b_out < B; b_out++) {
            float sum = 0.0f;
            for (size_t b_in = 0; b_in < B; b_in++) {
                sum += in_tensor[b_in * pixel_count + p] * tf->dense.kernel[b_in * B + b_out];
            }
            out_tensor[b_out * pixel_count + p] = sum;
        }
    }
}

void apply_spectral_synthesis(float* restrict out_tensor, const float* restrict in_tensor,
                              const SpectralTransform* tf, int H, int W) {
    apply_spectral_analysis(out_tensor, in_tensor, tf, H, W);
}


// apply_conv2d refactorizado: Paralelo por canal de salida + Tiling estacionario
// IMPORTANTE: Usa padding "same_zeros" de SignalConv2D (padding = kernel_size // 2)
// NO usa padding SAME de TensorFlow estándar
void apply_conv2d(float* restrict out_tensor, const float* restrict in_tensor, 
                  const ConvLayer* layer, int H_in, int W_in) {
    
    int kH = (int)layer->kH;
    int kW = (int)layer->kW;
    int C_in = (int)layer->C_in;
    int C_out = (int)layer->C_out;
    int stride = layer->stride;
    int H_out = (H_in + stride - 1) / stride;
    int W_out = (int)((W_in + stride - 1) / stride); // Cast explícito

    // SignalConv2D same_zeros padding: fijo = kernel_size // 2
    // Para kernel 5x5, padding = 2 en cada lado
    int pad_y = kH / 2;
    int pad_x = kW / 2;

    size_t in_plane_size = (size_t)H_in * W_in;
    size_t out_plane_size = (size_t)H_out * W_out;

    // Inicializar tensor de salida a cero para evitar residuos
    memset(out_tensor, 0, (size_t)C_out * H_out * W_out * sizeof(float));

    // Paralelizar sobre canales de salida
#ifdef _OPENMP
    #pragma omp parallel for if (C_out >= 4)
#endif
    for (int c_out = 0; c_out < C_out; c_out++) {
        float* p_out_plane = out_tensor + c_out * out_plane_size;
        float bias_val = (layer->has_bias && layer->bias) ? layer->bias[c_out] : 0.0f;
        
        // 1. Inicializar plano de salida con bias (bucle suficiente)
        for (size_t i = 0; i < out_plane_size; i++) p_out_plane[i] = bias_val;

        // 2. Procesamiento por bloques (tiles) sobre H_out
        // Tamaño de tile constante (ej: 8 filas). 8 filas × 512 = 16KB. Cabe en L1D (32KB).
        const int TILE_H = 8;
        
        for (int ty = 0; ty < H_out; ty += TILE_H) {
            int ty_end = ty + TILE_H;
            if (ty_end > H_out) ty_end = H_out;

            // Iterar sobre TODOS los canales de entrada para este tile de salida
            for (int c_in = 0; c_in < C_in; c_in++) {
                const float* p_in_plane = in_tensor + c_in * in_plane_size;

                // Iterar sobre el kernel
                for (int ky = 0; ky < kH; ky++) {
                    for (int kx = 0; kx < kW; kx++) {
                        // Obtener peso
                        size_t k_idx = (((size_t)ky * kW + kx) * C_in + c_in) * C_out + c_out;
                        float weight = layer->kernel[k_idx];
                        
                        // Preparar NEON
                        #if defined(USE_NEON)
                        float32x4_t w_vec = vdupq_n_f32(weight);
                        #endif
                        
                        // Bucles internos: iterar sobre las filas del tile
                        for (int y = ty; y < ty_end; y++) {
                             int iy = y * stride + ky - pad_y;
                             
                             // Comprobación de límites Y
                             if (iy < 0 || iy >= H_in) continue;
                             
                             // Punteros
                             const float* row_in = p_in_plane + iy * W_in;
                             float* row_out = p_out_plane + y * W_out;
                             
                             int ix_base = -pad_x + kx; // Offset base en X
                             int x = 0;
                             
                             // --- Bucles espaciales vectorizados ---
                             if (stride == 1) {
                                  #if defined(USE_NEON)
                                  for (; x <= W_out - 4; x += 4) {
                                      int ix0 = x + ix_base;
                                      if (ix0 >= 0 && (ix0 + 4) <= W_in) {
                                          float32x4_t in_vec = vld1q_f32(row_in + ix0);
                                          float32x4_t out_vec = vld1q_f32(row_out + x);
                                          out_vec = vmlaq_f32(out_vec, in_vec, w_vec);
                                          vst1q_f32(row_out + x, out_vec);
                                      } else {
                                          // Comprobación de límites por píxel (fallback)
                                          for (int k=0; k<4; k++) {
                                             int xx = x + k;
                                             int ixx = xx + ix_base;
                                             if (ixx >= 0 && ixx < W_in) row_out[xx] += row_in[ixx] * weight;
                                          }
                                      }
                                  }
                                  #endif
                             } else if (stride == 2) {
                                  #if defined(USE_NEON)
                                  for (; x <= W_out - 4; x += 4) {
                                      int ix0 = x * 2 + ix_base;
                                      if (ix0 >= 0 && (ix0 + 8) <= W_in) {
                                          float32x4x2_t in_deint = vld2q_f32(row_in + ix0);
                                          float32x4_t in_vec = in_deint.val[0];
                                          float32x4_t out_vec = vld1q_f32(row_out + x);
                                          out_vec = vmlaq_f32(out_vec, in_vec, w_vec);
                                          vst1q_f32(row_out + x, out_vec);
                                      } else {
                                          for (int k=0; k<4; k++) {
                                             int xx = x + k;
                                             int ixx = xx * 2 + ix_base;
                                             if (ixx >= 0 && ixx < W_in) row_out[xx] += row_in[ixx] * weight;
                                          }
                                      }
                                  }
                                  #endif
                             }
                             
                             // Resto escalar
                             for (; x < W_out; x++) {
                                 int ix = x * stride + ix_base;
                                 if (ix >= 0 && ix < W_in) row_out[x] += row_in[ix] * weight;
                             }
                        }
                    }
                }
            }
        }
    }
}

void apply_gdn(float* restrict out_tensor, const float* restrict in_tensor,
               const GDNLayer* layer, int H, int W) {
    // GDN con alpha=1, epsilon=1 (como usa tensorflow_compression por defecto)
    // Fórmula: y = x / (beta + sum_j(gamma[j,i] * |x_j|)) 
    // IMPORTANTE: TFC GDN usa |x| (valor absoluto), NO x^2
    // Y usa división directa, NO sqrt
    
    const size_t C = layer->C;
    const size_t plane_size = (size_t)H * (size_t)W;
    
    if (C == 0) return;

#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        float denom_stack[1024];
        float* denom = (C <= 1024) ? denom_stack : (float*)malloc(C * sizeof(float));
        if (!denom) {
            fprintf(stderr, "Error: memory allocation failed in apply_gdn\n");
        } else {
#ifdef _OPENMP
            #pragma omp for
#endif
            for (size_t p = 0; p < plane_size; ++p) {
                // Inicializar con Beta
                for (size_t i = 0; i < C; ++i) {
                    float val = layer->beta[i];
                    if (val < 1e-6f) val = 1e-6f;
                    denom[i] = val;
                }

                // Acumular gamma * |x| (valor absoluto, no cuadrado)
                for (size_t j = 0; j < C; ++j) {
                    float xj = in_tensor[j * plane_size + p];
                    float abs_xj = fabsf(xj);  // |x| en lugar de x^2
                    const float* gamma_row = layer->gamma + j * C;

                    // Vectorizar este producto interno si es posible (C=128 normalmente)
                    size_t i = 0;
                    #if defined(USE_NEON)
                    float32x4_t abs_vec = vdupq_n_f32(abs_xj);
                    for (; i + 3 < C; i += 4) {
                        float32x4_t g_vec = vld1q_f32(gamma_row + i);
                        float32x4_t d_vec = vld1q_f32(denom + i);
                        d_vec = vmlaq_f32(d_vec, g_vec, abs_vec);
                        vst1q_f32(denom + i, d_vec);
                    }
                    #endif
                    for (; i < C; ++i) denom[i] += gamma_row[i] * abs_xj;
                }

                // División final (sin sqrt porque epsilon=1)
                for (size_t i = 0; i < C; ++i) {
                    out_tensor[i * plane_size + p] = in_tensor[i * plane_size + p] / denom[i];
                }
            }
            if (denom != denom_stack) free(denom);
        }
    }
}

void apply_igdn(float* restrict out_tensor, const float* restrict in_tensor,
                const GDNLayer* layer, int H, int W) {
    const size_t C = layer->C;
    const size_t plane_size = (size_t)H * (size_t)W;

    if (C == 0) return;

#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        float denom_stack[1024];
        float* denom = (C <= 1024) ? denom_stack : (float*)malloc(C * sizeof(float));
        if (!denom) {
            fprintf(stderr, "Error: memory allocation failed in apply_igdn\n");
        } else {
#ifdef _OPENMP
            #pragma omp for
#endif
            for (size_t p = 0; p < plane_size; ++p) {
                for (size_t i = 0; i < C; ++i) {
                    float val = layer->beta[i];
                    if (val < 1e-6f) val = 1e-6f;
                    denom[i] = val;
                }

                for (size_t j = 0; j < C; ++j) {
                    float xj = in_tensor[j * plane_size + p];
                    float abs_xj = fabsf(xj);
                    const float* gamma_row = layer->gamma + j * C;
                    size_t i = 0;
                    #if defined(USE_NEON)
                    float32x4_t abs_vec = vdupq_n_f32(abs_xj);
                    for (; i + 3 < C; i += 4) {
                        float32x4_t g_vec = vld1q_f32(gamma_row + i);
                        float32x4_t d_vec = vld1q_f32(denom + i);
                        d_vec = vmlaq_f32(d_vec, g_vec, abs_vec);
                        vst1q_f32(denom + i, d_vec);
                    }
                    #endif
                    for (; i < C; ++i) denom[i] += gamma_row[i] * abs_xj;
                }

                for (size_t i = 0; i < C; ++i) {
                    out_tensor[i * plane_size + p] = in_tensor[i * plane_size + p] * denom[i];
                }
            }
            if (denom != denom_stack) free(denom);
        }
    }
}

void apply_conv2d_corrfalse(float* restrict out_tensor, const float* restrict in_tensor,
                            const ConvLayer* layer, int H_in, int W_in) {
    int kH = (int)layer->kH;
    int kW = (int)layer->kW;
    int C_in = (int)layer->C_in;
    int C_out = (int)layer->C_out;
    int stride = layer->stride;
    int H_out = (H_in + stride - 1) / stride;
    int W_out = (W_in + stride - 1) / stride;
    int pad_y = kH / 2;
    int pad_x = kW / 2;

    size_t in_plane_size = (size_t)H_in * W_in;
    size_t out_plane_size = (size_t)H_out * W_out;
    memset(out_tensor, 0, (size_t)C_out * out_plane_size * sizeof(float));

#ifdef _OPENMP
    #pragma omp parallel for if (C_out >= 4)
#endif
    for (int c_out = 0; c_out < C_out; ++c_out) {
        float* p_out = out_tensor + (size_t)c_out * out_plane_size;
        float bias_val = (layer->has_bias && layer->bias) ? layer->bias[c_out] : 0.0f;
        for (size_t i = 0; i < out_plane_size; ++i) p_out[i] = bias_val;

        const int TILE_H = 8;
        for (int ty = 0; ty < H_out; ty += TILE_H) {
            int ty_end = ty + TILE_H;
            if (ty_end > H_out) ty_end = H_out;

            for (int c_in = 0; c_in < C_in; ++c_in) {
                const float* p_in = in_tensor + (size_t)c_in * in_plane_size;
                for (int ky = 0; ky < kH; ++ky) {
                    int fky = kH - 1 - ky;
                    for (int kx = 0; kx < kW; ++kx) {
                        int fkx = kW - 1 - kx;
                        size_t k_idx = (((size_t)fky * kW + fkx) * (size_t)C_in + (size_t)c_in) * (size_t)C_out + (size_t)c_out;
                        float weight = layer->kernel[k_idx];

                        #if defined(USE_NEON)
                        float32x4_t w_vec = vdupq_n_f32(weight);
                        #endif

                        for (int y = ty; y < ty_end; ++y) {
                            int iy = y * stride + ky - pad_y;
                            if (iy < 0 || iy >= H_in) continue;

                            const float* row_in = p_in + (size_t)iy * (size_t)W_in;
                            float* row_out = p_out + (size_t)y * (size_t)W_out;
                            int ix_base = -pad_x + kx;
                            int x = 0;

                            if (stride == 1) {
                                #if defined(USE_NEON)
                                for (; x <= W_out - 4; x += 4) {
                                    int ix0 = x + ix_base;
                                    if (ix0 >= 0 && (ix0 + 4) <= W_in) {
                                        float32x4_t in_vec = vld1q_f32(row_in + ix0);
                                        float32x4_t out_vec = vld1q_f32(row_out + x);
                                        out_vec = vmlaq_f32(out_vec, in_vec, w_vec);
                                        vst1q_f32(row_out + x, out_vec);
                                    } else {
                                        for (int k = 0; k < 4; ++k) {
                                            int xx = x + k;
                                            int ixx = xx + ix_base;
                                            if (ixx >= 0 && ixx < W_in) row_out[xx] += row_in[ixx] * weight;
                                        }
                                    }
                                }
                                #endif
                            } else if (stride == 2) {
                                #if defined(USE_NEON)
                                for (; x <= W_out - 4; x += 4) {
                                    int ix0 = x * 2 + ix_base;
                                    if (ix0 >= 0 && (ix0 + 8) <= W_in) {
                                        float32x4x2_t in_deint = vld2q_f32(row_in + ix0);
                                        float32x4_t in_vec = in_deint.val[0];
                                        float32x4_t out_vec = vld1q_f32(row_out + x);
                                        out_vec = vmlaq_f32(out_vec, in_vec, w_vec);
                                        vst1q_f32(row_out + x, out_vec);
                                    } else {
                                        for (int k = 0; k < 4; ++k) {
                                            int xx = x + k;
                                            int ixx = xx * 2 + ix_base;
                                            if (ixx >= 0 && ixx < W_in) row_out[xx] += row_in[ixx] * weight;
                                        }
                                    }
                                }
                                #endif
                            }

                            for (; x < W_out; ++x) {
                                int ix = x * stride + ix_base;
                                if (ix >= 0 && ix < W_in) row_out[x] += row_in[ix] * weight;
                            }
                        }
                    }
                }
            }
        }
    }
}

void apply_depth_to_space_2x(float* restrict out_tensor, const float* restrict in_tensor,
                             int C_in, int H_in, int W_in) {
    if (C_in % 4 != 0) {
        fprintf(stderr, "Error: apply_depth_to_space_2x requiere C_in múltiplo de 4 (C_in=%d)\n", C_in);
        return;
    }
    int C_out = C_in / 4;
    int H_out = H_in * 2;
    int W_out = W_in * 2;
    size_t in_plane = (size_t)H_in * W_in;
    size_t out_plane = (size_t)H_out * W_out;

#ifdef _OPENMP
    #pragma omp parallel for if (C_out >= 4)
#endif
    for (int c = 0; c < C_out; ++c) {
        // tf.nn.depth_to_space (NHWC, block=2):
        // out[..., c] toma in[..., c + C_out*{0,1,2,3}] para [00,01,10,11].
        size_t c0 = (size_t)(c + C_out * 0) * in_plane;
        size_t c1 = (size_t)(c + C_out * 1) * in_plane;
        size_t c2 = (size_t)(c + C_out * 2) * in_plane;
        size_t c3 = (size_t)(c + C_out * 3) * in_plane;
        size_t co = (size_t)c * out_plane;

        for (int y = 0; y < H_in; ++y) {
            int oy = y * 2;
            for (int x = 0; x < W_in; ++x) {
                int ox = x * 2;
                size_t in_idx = (size_t)y * W_in + x;
                out_tensor[co + (size_t)oy * W_out + ox] = in_tensor[c0 + in_idx];
                out_tensor[co + (size_t)oy * W_out + (ox + 1)] = in_tensor[c1 + in_idx];
                out_tensor[co + (size_t)(oy + 1) * W_out + ox] = in_tensor[c2 + in_idx];
                out_tensor[co + (size_t)(oy + 1) * W_out + (ox + 1)] = in_tensor[c3 + in_idx];
            }
        }
    }
}
