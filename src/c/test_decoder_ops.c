#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "sorteny_model.h"

static int almost_equal(float a, float b, float tol) {
    return fabsf(a - b) <= tol;
}

static int test_depth_to_space(void) {
    // C_in=8, H=W=1 -> C_out=2, H=W=2
    float in[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    float out[8] = {0};
    float ref[8] = {
        // tf.nn.depth_to_space block=2:
        // canal 0 (2x2): [0,2;4,6]
        0, 2, 4, 6,
        // canal 1 (2x2): [1,3;5,7]
        1, 3, 5, 7
    };

    apply_depth_to_space_2x(out, in, 8, 1, 1);
    for (int i = 0; i < 8; ++i) {
        if (!almost_equal(out[i], ref[i], 1e-7f)) {
            fprintf(stderr, "depth_to_space mismatch at %d: got=%.6f ref=%.6f\n", i, out[i], ref[i]);
            return -1;
        }
    }
    return 0;
}

static int test_depth_to_space_multi_pixel(void) {
    // C_in=8, H=W=2 -> C_out=2, H=W=4.
    float in[32];
    float out[32] = {0};
    float ref[32] = {0};
    for (int c = 0; c < 8; ++c) {
        for (int p = 0; p < 4; ++p) {
            in[c * 4 + p] = (float)(c * 10 + p);
        }
    }

    apply_depth_to_space_2x(out, in, 8, 2, 2);

    const int C_out = 2;
    const int H_in = 2, W_in = 2;
    const int H_out = 4, W_out = 4;
    const int in_plane = H_in * W_in;
    const int out_plane = H_out * W_out;
    for (int c = 0; c < C_out; ++c) {
        int c0 = c + C_out * 0;
        int c1 = c + C_out * 1;
        int c2 = c + C_out * 2;
        int c3 = c + C_out * 3;
        for (int y = 0; y < H_in; ++y) {
            for (int x = 0; x < W_in; ++x) {
                int in_idx = y * W_in + x;
                int oy = y * 2;
                int ox = x * 2;
                ref[c * out_plane + oy * W_out + ox] = in[c0 * in_plane + in_idx];
                ref[c * out_plane + oy * W_out + ox + 1] = in[c1 * in_plane + in_idx];
                ref[c * out_plane + (oy + 1) * W_out + ox] = in[c2 * in_plane + in_idx];
                ref[c * out_plane + (oy + 1) * W_out + ox + 1] = in[c3 * in_plane + in_idx];
            }
        }
    }

    for (int i = 0; i < 32; ++i) {
        if (!almost_equal(out[i], ref[i], 1e-7f)) {
            fprintf(stderr, "depth_to_space multi mismatch at %d: got=%.6f ref=%.6f\n", i, out[i], ref[i]);
            return -1;
        }
    }
    return 0;
}

static void conv_corrfalse_reference(float* out, const float* in, const float* k, const float* b,
                                     int H, int W, int kH, int kW) {
    int pad_y = kH / 2, pad_x = kW / 2;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float acc = b ? b[0] : 0.0f;
            int in_y_base = y - pad_y;
            int in_x_base = x - pad_x;
            for (int ky = 0; ky < kH; ++ky) {
                int iy = in_y_base + ky;
                if (iy < 0 || iy >= H) continue;
                int fky = kH - 1 - ky;
                for (int kx = 0; kx < kW; ++kx) {
                    int ix = in_x_base + kx;
                    if (ix < 0 || ix >= W) continue;
                    int fkx = kW - 1 - kx;
                    acc += in[iy * W + ix] * k[fky * kW + fkx];
                }
            }
            out[y * W + x] = acc;
        }
    }
}

static int test_conv_corrfalse(void) {
    // Caso simple: 1 canal entrada / 1 salida / stride=1 / same_zeros.
    float in[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    float kernel[9] = {
         1, 0, -1,
         2, 0, -2,
         1, 0, -1
    };
    float bias[1] = {0.5f};

    ConvLayer layer;
    memset(&layer, 0, sizeof(layer));
    layer.kernel = kernel;
    layer.bias = bias;
    layer.kH = 3;
    layer.kW = 3;
    layer.C_in = 1;
    layer.C_out = 1;
    layer.stride = 1;
    layer.has_bias = 1;

    float out[9] = {0};
    float ref[9] = {0};
    apply_conv2d_corrfalse(out, in, &layer, 3, 3);
    conv_corrfalse_reference(ref, in, kernel, bias, 3, 3, 3, 3);

    for (int i = 0; i < 9; ++i) {
        if (!almost_equal(out[i], ref[i], 1e-6f)) {
            fprintf(stderr, "conv_corrfalse mismatch at %d: got=%.6f ref=%.6f\n", i, out[i], ref[i]);
            return -1;
        }
    }
    return 0;
}

static int test_spectral_analysis(void) {
    // 3 bandas, 2 píxeles. La matriz kernel se indexa b_in * B + b_out.
    float in[6] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f
    };
    float kernel[9] = {
        1.0f, 0.0f, 2.0f,
        0.5f, 1.0f, 0.0f,
        0.0f, 1.5f, 1.0f
    };
    float out[6] = {0};
    float ref[6] = {0};
    SpectralTransform tf;
    memset(&tf, 0, sizeof(tf));
    tf.dense.kernel = kernel;
    tf.dense.C_in = 3;
    tf.dense.C_out = 3;

    apply_spectral_analysis(out, in, &tf, 1, 2);

    for (int p = 0; p < 2; ++p) {
        for (int b_out = 0; b_out < 3; ++b_out) {
            float sum = 0.0f;
            for (int b_in = 0; b_in < 3; ++b_in) {
                sum += in[b_in * 2 + p] * kernel[b_in * 3 + b_out];
            }
            ref[b_out * 2 + p] = sum;
        }
    }

    for (int i = 0; i < 6; ++i) {
        if (!almost_equal(out[i], ref[i], 1e-6f)) {
            fprintf(stderr, "spectral_analysis mismatch at %d: got=%.6f ref=%.6f\n", i, out[i], ref[i]);
            return -1;
        }
    }
    return 0;
}

static int test_gdn(void) {
    // 2 canales, 2 píxeles por canal.
    float in[4] = {2.0f, 4.0f, -1.0f, 3.0f};
    float out[4] = {0};
    float beta[2] = {1.0f, 2.0f};
    // gamma[j,i]
    float gamma[4] = {
        0.5f, 0.1f,
        0.2f, 0.4f
    };
    GDNLayer gdn;
    memset(&gdn, 0, sizeof(gdn));
    gdn.beta = beta;
    gdn.gamma = gamma;
    gdn.C = 2;
    gdn.epsilon = 1.0f;

    apply_gdn(out, in, &gdn, 1, 2);

    float ref[4] = {
        2.0f / 2.2f,
        4.0f / 3.6f,
        -1.0f / 2.6f,
        3.0f / 3.6f
    };
    for (int i = 0; i < 4; ++i) {
        if (!almost_equal(out[i], ref[i], 1e-6f)) {
            fprintf(stderr, "gdn mismatch at %d: got=%.6f ref=%.6f\n", i, out[i], ref[i]);
            return -1;
        }
    }
    return 0;
}

static int test_igdn(void) {
    // 2 canales, 1 píxel.
    float in[2] = {2.0f, -1.0f};
    float out[2] = {0};
    float beta[2] = {1.0f, 2.0f};
    // gamma[j,i]
    float gamma[4] = {
        0.5f, 0.1f,
        0.2f, 0.4f
    };
    GDNLayer igdn;
    memset(&igdn, 0, sizeof(igdn));
    igdn.beta = beta;
    igdn.gamma = gamma;
    igdn.C = 2;
    igdn.epsilon = 1.0f;

    apply_igdn(out, in, &igdn, 1, 1);

    // y0 = x0 * (1 + 0.5*|x0| + 0.2*|x1|) = 2 * 2.2 = 4.4
    // y1 = x1 * (2 + 0.1*|x0| + 0.4*|x1|) = -1 * 2.6 = -2.6
    if (!almost_equal(out[0], 4.4f, 1e-6f) || !almost_equal(out[1], -2.6f, 1e-6f)) {
        fprintf(stderr, "igdn mismatch: got=[%.6f, %.6f] ref=[4.4, -2.6]\n", out[0], out[1]);
        return -1;
    }
    return 0;
}

int main(void) {
    int rc = 0;
    rc |= test_depth_to_space();
    rc |= test_depth_to_space_multi_pixel();
    rc |= test_conv_corrfalse();
    rc |= test_spectral_analysis();
    rc |= test_gdn();
    rc |= test_igdn();

    if (rc == 0) {
        printf("OK: decoder ops tests passed.\n");
        return 0;
    }
    fprintf(stderr, "FAIL: decoder ops tests failed.\n");
    return 1;
}
