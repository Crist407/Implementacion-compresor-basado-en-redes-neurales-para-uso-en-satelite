#define _POSIX_C_SOURCE 200112L
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_WIDTH 32
#define DEFAULT_HEIGHT 32
#define MAX_U16 65535.0

typedef struct {
    int block_y;
    int block_x;
    double c0;
    double c1;
    double r2;
    int valid;
    int q_baseline;
    int q_min;
    int q_max;
    double max_lambda;
    double mod_a;
    double mod_b;
    double mse_at_baseline;
    int loaded;
} FQBlockCalib;

typedef enum {
    TARGET_NONE = 0,
    TARGET_MSE,
    TARGET_PSNR,
    TARGET_FROM_Q,
    TARGET_ADAPTIVE_DIFFICULTY
} TargetMode;

typedef struct {
    const char* calibration_path;
    const char* output_qmap_path;
    const char* summary_tsv_path;
    int width;
    int height;
    TargetMode mode;
    double target_value;
    int target_q;
    int adaptive_q_mean;
    double adaptive_strength;
} Options;

typedef struct {
    int reachable;
    int too_strict;
    int too_relaxed;
    int invalid;
    int adaptive_budget;
    int clamped;
    int fallback;
    double predicted_mse_sum;
    double best_mse_sum;
    double worst_mse_sum;
} FeasibilitySummary;

static void usage(const char* argv0) {
    fprintf(stderr,
        "Uso: %s --calibration fq_calibration.tsv --output-qmap qmap.bin "
        "(--target-mse MSE | --target-psnr PSNR | --target-from-q Q | --adaptive-difficulty) [opciones]\n\n"
        "Opciones:\n"
        "  --width N                 Anchura del Q-map (default: 32)\n"
        "  --height N                Altura del Q-map (default: 32)\n"
        "  --summary-tsv path        Guarda decision por bloque en TSV\n"
        "  --q-mean Q                Presupuesto medio Q para --adaptive-difficulty (default: 204)\n"
        "  --adaptive-strength S     Intensidad adaptativa en Q por sigma log-MSE (default: 16)\n"
        "\n"
        "Modelo: MSE ~= c0 + c1 / (mod_a*lambda + mod_b)^2\n",
        argv0);
}

static int parse_int(const char* s, int* out) {
    char* end = NULL;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v < -2147483647L || v > 2147483647L) {
        return -1;
    }
    *out = (int)v;
    return 0;
}

static int parse_double(const char* s, double* out) {
    char* end = NULL;
    errno = 0;
    double v = strtod(s, &end);
    if (errno != 0 || end == s || *end != '\0' || !isfinite(v)) {
        return -1;
    }
    *out = v;
    return 0;
}

static int set_target(Options* opt, TargetMode mode, const char* value) {
    if (opt->mode != TARGET_NONE) {
        fprintf(stderr, "Error: usa solo un objetivo de calidad.\n");
        return -1;
    }
    opt->mode = mode;
    if (mode == TARGET_FROM_Q) {
        if (parse_int(value, &opt->target_q) != 0 || opt->target_q < 0 || opt->target_q > 255) {
            fprintf(stderr, "Error: --target-from-q debe estar en [0,255].\n");
            return -1;
        }
    } else {
        if (parse_double(value, &opt->target_value) != 0 || opt->target_value <= 0.0) {
            fprintf(stderr, "Error: objetivo numerico invalido.\n");
            return -1;
        }
    }
    return 0;
}

static int parse_args(int argc, char** argv, Options* opt) {
    memset(opt, 0, sizeof(*opt));
    opt->width = DEFAULT_WIDTH;
    opt->height = DEFAULT_HEIGHT;
    opt->mode = TARGET_NONE;
    opt->adaptive_q_mean = 204;
    opt->adaptive_strength = 16.0;

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (strcmp(a, "--calibration") == 0 && i + 1 < argc) {
            opt->calibration_path = argv[++i];
        } else if (strcmp(a, "--output-qmap") == 0 && i + 1 < argc) {
            opt->output_qmap_path = argv[++i];
        } else if (strcmp(a, "--summary-tsv") == 0 && i + 1 < argc) {
            opt->summary_tsv_path = argv[++i];
        } else if (strcmp(a, "--width") == 0 && i + 1 < argc) {
            if (parse_int(argv[++i], &opt->width) != 0 || opt->width <= 0) return -1;
        } else if (strcmp(a, "--height") == 0 && i + 1 < argc) {
            if (parse_int(argv[++i], &opt->height) != 0 || opt->height <= 0) return -1;
        } else if (strcmp(a, "--target-mse") == 0 && i + 1 < argc) {
            if (set_target(opt, TARGET_MSE, argv[++i]) != 0) return -1;
        } else if (strcmp(a, "--target-psnr") == 0 && i + 1 < argc) {
            if (set_target(opt, TARGET_PSNR, argv[++i]) != 0) return -1;
        } else if (strcmp(a, "--target-from-q") == 0 && i + 1 < argc) {
            if (set_target(opt, TARGET_FROM_Q, argv[++i]) != 0) return -1;
        } else if (strcmp(a, "--adaptive-difficulty") == 0) {
            if (opt->mode != TARGET_NONE) {
                fprintf(stderr, "Error: usa solo un objetivo de calidad.\n");
                return -1;
            }
            opt->mode = TARGET_ADAPTIVE_DIFFICULTY;
        } else if (strcmp(a, "--q-mean") == 0 && i + 1 < argc) {
            if (parse_int(argv[++i], &opt->adaptive_q_mean) != 0 ||
                opt->adaptive_q_mean < 0 || opt->adaptive_q_mean > 255) {
                fprintf(stderr, "Error: --q-mean debe estar en [0,255].\n");
                return -1;
            }
        } else if (strcmp(a, "--adaptive-strength") == 0 && i + 1 < argc) {
            if (parse_double(argv[++i], &opt->adaptive_strength) != 0 || opt->adaptive_strength < 0.0) {
                fprintf(stderr, "Error: --adaptive-strength debe ser >= 0.\n");
                return -1;
            }
        } else if (strcmp(a, "-h") == 0 || strcmp(a, "--help") == 0) {
            usage(argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "Error: argumento desconocido o incompleto: %s\n", a);
            return -1;
        }
    }

    if (!opt->calibration_path || !opt->output_qmap_path || opt->mode == TARGET_NONE) {
        return -1;
    }
    return 0;
}

static double psnr_to_mse(double psnr_db) {
    return (MAX_U16 * MAX_U16) / pow(10.0, psnr_db / 10.0);
}

static int clamp_int(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static double lambda_from_q(int q, double max_lambda) {
    return ((double)q / 255.0) * max_lambda;
}

static double mod_from_lambda(const FQBlockCalib* c, double lambda) {
    return c->mod_a * lambda + c->mod_b;
}

static double predicted_mse_for_q(const FQBlockCalib* c, int q) {
    double lambda = lambda_from_q(q, c->max_lambda);
    double m = mod_from_lambda(c, lambda);
    if (m <= 0.0) m = 1e-12;
    return c->c0 + (c->c1 / (m * m));
}

static const char* feasibility_for_target(const FQBlockCalib* c, double target_mse) {
    if (!c->valid || c->c1 <= 0.0 || c->mod_a <= 0.0 || c->max_lambda <= 0.0) {
        return "invalid";
    }
    double best_mse = predicted_mse_for_q(c, c->q_max);
    double worst_mse = predicted_mse_for_q(c, c->q_min);
    if (target_mse < best_mse) {
        return "too_strict";
    }
    if (target_mse > worst_mse) {
        return "too_relaxed";
    }
    return "reachable";
}

static int select_q_for_target(const FQBlockCalib* c, TargetMode mode, double target_mse, int target_q, const char** reason) {
    if (!c->valid || c->c1 <= 0.0 || c->mod_a <= 0.0 || c->max_lambda <= 0.0) {
        *reason = "fallback_invalid_calibration";
        return clamp_int(c->q_baseline, c->q_min, c->q_max);
    }

    double local_target = target_mse;
    if (mode == TARGET_FROM_Q) {
        int q = clamp_int(target_q, c->q_min, c->q_max);
        local_target = predicted_mse_for_q(c, q);
    }

    if (!(local_target > c->c0)) {
        *reason = "clamped_high_quality_floor";
        return c->q_max;
    }

    double target_mod = sqrt(c->c1 / (local_target - c->c0));
    double lambda = (target_mod - c->mod_b) / c->mod_a;
    double q_float = (lambda / c->max_lambda) * 255.0;
    if (!isfinite(q_float)) {
        *reason = "fallback_nonfinite";
        return clamp_int(c->q_baseline, c->q_min, c->q_max);
    }

    int q_raw = (int)lround(q_float);
    int q = clamp_int(q_raw, c->q_min, c->q_max);
    if (q != q_raw) {
        *reason = (q_raw < c->q_min) ? "clamped_low_q" : "clamped_high_q";
    } else {
        *reason = "model";
    }
    return q;
}

static void log_mse_stats(const FQBlockCalib* blocks, int count, double* mean_out, double* std_out) {
    double sum = 0.0;
    double sum2 = 0.0;
    int n = 0;
    for (int i = 0; i < count; ++i) {
        if (!blocks[i].valid || blocks[i].mse_at_baseline <= 0.0) continue;
        double v = log(blocks[i].mse_at_baseline);
        sum += v;
        sum2 += v * v;
        n++;
    }
    if (n <= 0) {
        *mean_out = 0.0;
        *std_out = 1.0;
        return;
    }
    double mean = sum / (double)n;
    double var = (sum2 / (double)n) - (mean * mean);
    if (var < 1e-12) var = 1e-12;
    *mean_out = mean;
    *std_out = sqrt(var);
}

static int select_q_adaptive_difficulty(
    const FQBlockCalib* c,
    int q_mean,
    double strength,
    double log_mean,
    double log_std,
    const char** reason
) {
    if (!c->valid || c->mse_at_baseline <= 0.0 || log_std <= 0.0) {
        *reason = "fallback_invalid_calibration";
        return clamp_int(c->q_baseline, c->q_min, c->q_max);
    }
    double z = (log(c->mse_at_baseline) - log_mean) / log_std;
    double q_float = (double)q_mean + (strength * z);
    int q_raw = (int)lround(q_float);
    int q = clamp_int(q_raw, c->q_min, c->q_max);
    if (q != q_raw) {
        *reason = (q_raw < c->q_min) ? "adaptive_clamped_low_q" : "adaptive_clamped_high_q";
    } else {
        *reason = "adaptive_difficulty";
    }
    return q;
}

static int load_calibration(const char* path, FQBlockCalib* blocks, int width, int height) {
    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Error: no se pudo abrir calibracion '%s'\n", path);
        return -1;
    }

    char line[4096];
    size_t line_no = 0;
    int loaded = 0;
    while (fgets(line, sizeof(line), f)) {
        line_no++;
        if (line[0] == '#' || line[0] == '\n' || strncmp(line, "block_y", 7) == 0) {
            continue;
        }

        FQBlockCalib c;
        memset(&c, 0, sizeof(c));
        int n = sscanf(
            line,
            "%d\t%d\t%lf\t%lf\t%lf\t%d\t%d\t%d\t%d\t%lf\t%lf\t%lf\t%lf",
            &c.block_y,
            &c.block_x,
            &c.c0,
            &c.c1,
            &c.r2,
            &c.valid,
            &c.q_baseline,
            &c.q_min,
            &c.q_max,
            &c.max_lambda,
            &c.mod_a,
            &c.mod_b,
            &c.mse_at_baseline
        );
        if (n != 13) {
            fprintf(stderr, "Error: linea de calibracion invalida %zu\n", line_no);
            fclose(f);
            return -1;
        }
        if (c.block_y < 0 || c.block_y >= height || c.block_x < 0 || c.block_x >= width) {
            fprintf(stderr, "Error: bloque fuera de rango en linea %zu\n", line_no);
            fclose(f);
            return -1;
        }
        if (c.q_min < 0) c.q_min = 0;
        if (c.q_max > 255) c.q_max = 255;
        if (c.q_min > c.q_max) {
            fprintf(stderr, "Error: q_min > q_max en linea %zu\n", line_no);
            fclose(f);
            return -1;
        }
        c.q_baseline = clamp_int(c.q_baseline, c.q_min, c.q_max);
        c.loaded = 1;
        blocks[c.block_y * width + c.block_x] = c;
        loaded++;
    }
    fclose(f);

    int expected = width * height;
    if (loaded != expected) {
        fprintf(stderr, "Error: calibracion contiene %d bloques, esperado %d\n", loaded, expected);
        return -1;
    }
    for (int i = 0; i < expected; ++i) {
        if (!blocks[i].loaded) {
            fprintf(stderr, "Error: falta bloque de calibracion index %d\n", i);
            return -1;
        }
    }
    return 0;
}

static int write_qmap(const char* path, const uint8_t* qmap, size_t size) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "Error: no se pudo abrir salida Q-map '%s'\n", path);
        return -1;
    }
    size_t n = fwrite(qmap, sizeof(uint8_t), size, f);
    fclose(f);
    if (n != size) {
        fprintf(stderr, "Error: escritura incompleta de Q-map\n");
        return -1;
    }
    return 0;
}

static void print_q_summary(const uint8_t* qmap, int count) {
    int hist[256] = {0};
    int q_min = 255;
    int q_max = 0;
    long sum = 0;
    for (int i = 0; i < count; ++i) {
        int q = qmap[i];
        hist[q]++;
        if (q < q_min) q_min = q;
        if (q > q_max) q_max = q;
        sum += q;
    }
    int unique = 0;
    for (int q = 0; q < 256; ++q) {
        if (hist[q]) unique++;
    }
    printf("Q-map generado: blocks=%d unique_Q=%d q_min=%d q_max=%d q_mean=%.4f\n",
           count, unique, q_min, q_max, (double)sum / (double)count);
    printf("Q usados:");
    for (int q = 0; q < 256; ++q) {
        if (hist[q]) printf(" %d:%d", q, hist[q]);
    }
    printf("\n");
}

int main(int argc, char** argv) {
    Options opt;
    if (parse_args(argc, argv, &opt) != 0) {
        usage(argv[0]);
        return 2;
    }

    int count = opt.width * opt.height;
    FQBlockCalib* blocks = (FQBlockCalib*)calloc((size_t)count, sizeof(FQBlockCalib));
    uint8_t* qmap = (uint8_t*)malloc((size_t)count);
    if (!blocks || !qmap) {
        fprintf(stderr, "Error: memoria insuficiente\n");
        free(blocks);
        free(qmap);
        return 1;
    }

    if (load_calibration(opt.calibration_path, blocks, opt.width, opt.height) != 0) {
        free(blocks);
        free(qmap);
        return 1;
    }

    double target_mse = 0.0;
    if (opt.mode == TARGET_MSE) {
        target_mse = opt.target_value;
    } else if (opt.mode == TARGET_PSNR) {
        target_mse = psnr_to_mse(opt.target_value);
    }
    double log_mean = 0.0;
    double log_std = 1.0;
    if (opt.mode == TARGET_ADAPTIVE_DIFFICULTY) {
        log_mse_stats(blocks, count, &log_mean, &log_std);
    }

    FILE* summary = NULL;
    if (opt.summary_tsv_path) {
        summary = fopen(opt.summary_tsv_path, "w");
        if (!summary) {
            fprintf(stderr, "Error: no se pudo abrir summary TSV '%s'\n", opt.summary_tsv_path);
            free(blocks);
            free(qmap);
            return 1;
        }
        fprintf(summary, "block_y\tblock_x\tq\tpredicted_mse\tbest_mse_qmax\tworst_mse_qmin\tviability\treason\tc0\tc1\tr2\n");
    }

    FeasibilitySummary fs;
    memset(&fs, 0, sizeof(fs));
    for (int i = 0; i < count; ++i) {
        const char* reason = NULL;
        double local_target_mse = target_mse;
        const char* viability = NULL;
        if (opt.mode == TARGET_FROM_Q) {
            int q_ref = clamp_int(opt.target_q, blocks[i].q_min, blocks[i].q_max);
            local_target_mse = predicted_mse_for_q(&blocks[i], q_ref);
            viability = feasibility_for_target(&blocks[i], local_target_mse);
        } else if (opt.mode == TARGET_ADAPTIVE_DIFFICULTY) {
            local_target_mse = blocks[i].mse_at_baseline;
            viability = blocks[i].valid ? "adaptive_budget" : "invalid";
        } else {
            viability = feasibility_for_target(&blocks[i], local_target_mse);
        }
        int q;
        if (opt.mode == TARGET_ADAPTIVE_DIFFICULTY) {
            q = select_q_adaptive_difficulty(
                &blocks[i],
                opt.adaptive_q_mean,
                opt.adaptive_strength,
                log_mean,
                log_std,
                &reason
            );
        } else {
            q = select_q_for_target(&blocks[i], opt.mode, target_mse, opt.target_q, &reason);
        }
        double predicted = predicted_mse_for_q(&blocks[i], q);
        double best_mse = predicted_mse_for_q(&blocks[i], blocks[i].q_max);
        double worst_mse = predicted_mse_for_q(&blocks[i], blocks[i].q_min);

        qmap[i] = (uint8_t)q;
        if (strcmp(viability, "reachable") == 0) fs.reachable++;
        else if (strcmp(viability, "too_strict") == 0) fs.too_strict++;
        else if (strcmp(viability, "too_relaxed") == 0) fs.too_relaxed++;
        else if (strcmp(viability, "adaptive_budget") == 0) fs.adaptive_budget++;
        else fs.invalid++;
        if (strncmp(reason, "clamped", 7) == 0) fs.clamped++;
        if (strncmp(reason, "fallback", 8) == 0) fs.fallback++;
        fs.predicted_mse_sum += predicted;
        fs.best_mse_sum += best_mse;
        fs.worst_mse_sum += worst_mse;

        if (summary) {
            fprintf(summary, "%d\t%d\t%d\t%.9g\t%.9g\t%.9g\t%s\t%s\t%.9g\t%.9g\t%.9g\n",
                    blocks[i].block_y,
                    blocks[i].block_x,
                    q,
                    predicted,
                    best_mse,
                    worst_mse,
                    viability,
                    reason,
                    blocks[i].c0,
                    blocks[i].c1,
                    blocks[i].r2);
        }
    }

    if (summary) fclose(summary);

    if (write_qmap(opt.output_qmap_path, qmap, (size_t)count) != 0) {
        free(blocks);
        free(qmap);
        return 1;
    }

    if (opt.mode == TARGET_MSE) {
        printf("Objetivo: MSE %.9g\n", target_mse);
    } else if (opt.mode == TARGET_PSNR) {
        printf("Objetivo: PSNR %.6f dB (MSE %.9g)\n", opt.target_value, target_mse);
    } else if (opt.mode == TARGET_ADAPTIVE_DIFFICULTY) {
        printf("Objetivo: adaptive_difficulty q_mean=%d strength=%.6f log_mse_mean=%.6f log_mse_std=%.6f\n",
               opt.adaptive_q_mean, opt.adaptive_strength, log_mean, log_std);
    } else {
        printf("Objetivo: equivalente modelo a Q=%d\n", opt.target_q);
    }
    print_q_summary(qmap, count);
    printf("Viabilidad: reachable=%d too_strict=%d too_relaxed=%d adaptive_budget=%d invalid=%d\n",
           fs.reachable, fs.too_strict, fs.too_relaxed, fs.adaptive_budget, fs.invalid);
    printf("Prediccion media: selected_mse=%.6f best_mse_qmax=%.6f worst_mse_qmin=%.6f\n",
           fs.predicted_mse_sum / (double)count,
           fs.best_mse_sum / (double)count,
           fs.worst_mse_sum / (double)count);
    printf("Bloques clampados=%d fallback=%d salida=%s\n", fs.clamped, fs.fallback, opt.output_qmap_path);

    free(blocks);
    free(qmap);
    return 0;
}
