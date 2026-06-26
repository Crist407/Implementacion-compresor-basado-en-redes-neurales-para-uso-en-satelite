#define _POSIX_C_SOURCE 200112L
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_WIDTH 32
#define DEFAULT_HEIGHT 32
#define DEFAULT_MAX_LAMBDA 0.125
#define MAX_U16 65535.0
#define MAX_POINTS 512

typedef enum {
    TARGET_NONE = 0,
    TARGET_PSNR,
    TARGET_MSE
} TargetMode;

typedef struct {
    int q;
    double psnr_db;
    double mse;
} CalibPoint;

typedef struct {
    const char* calibration_path;
    const char* output_qmap_path;
    const char* summary_tsv_path;
    int width;
    int height;
    double max_lambda;
    TargetMode mode;
    double target_value;
} Options;

static void usage(const char* argv0) {
    fprintf(stderr,
        "Uso: %s --calibration global_q_quality.tsv --output-qmap qmap.raw "
        "(--target-psnr X | --target-mse X) [opciones]\n\n"
        "Opciones:\n"
        "  --width N             Anchura del Q-map (default: 32)\n"
        "  --height N            Altura del Q-map (default: 32)\n"
        "  --max-lambda X        Lambda maxima para resumen (default: 0.125)\n"
        "  --summary-tsv path    Guarda decision global en TSV\n\n"
        "La calibracion debe contener columnas q, psnr_db y mse. Se aceptan TSV,\n"
        "CSV o espacios como separadores. Este modo genera una Q constante para\n"
        "toda la imagen; no sustituye al target local por bloque de sorteny_fq_qmap.\n",
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

static int clamp_int(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static int set_target(Options* opt, TargetMode mode, const char* value) {
    if (opt->mode != TARGET_NONE) {
        fprintf(stderr, "Error: usa solo uno de --target-psnr o --target-mse.\n");
        return -1;
    }
    if (parse_double(value, &opt->target_value) != 0 || opt->target_value <= 0.0) {
        fprintf(stderr, "Error: objetivo invalido.\n");
        return -1;
    }
    opt->mode = mode;
    return 0;
}

static int parse_args(int argc, char** argv, Options* opt) {
    memset(opt, 0, sizeof(*opt));
    opt->width = DEFAULT_WIDTH;
    opt->height = DEFAULT_HEIGHT;
    opt->max_lambda = DEFAULT_MAX_LAMBDA;
    opt->mode = TARGET_NONE;

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
        } else if (strcmp(a, "--max-lambda") == 0 && i + 1 < argc) {
            if (parse_double(argv[++i], &opt->max_lambda) != 0 || opt->max_lambda <= 0.0) return -1;
        } else if (strcmp(a, "--target-psnr") == 0 && i + 1 < argc) {
            if (set_target(opt, TARGET_PSNR, argv[++i]) != 0) return -1;
        } else if (strcmp(a, "--target-mse") == 0 && i + 1 < argc) {
            if (set_target(opt, TARGET_MSE, argv[++i]) != 0) return -1;
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

static void trim_right(char* s) {
    size_t n = strlen(s);
    while (n > 0) {
        unsigned char c = (unsigned char)s[n - 1];
        if (c != '\n' && c != '\r' && c != ' ' && c != '\t') break;
        s[n - 1] = '\0';
        n--;
    }
}

static int cmp_q(const void* a, const void* b) {
    const CalibPoint* pa = (const CalibPoint*)a;
    const CalibPoint* pb = (const CalibPoint*)b;
    return (pa->q > pb->q) - (pa->q < pb->q);
}

static int load_calibration(const char* path, CalibPoint* points, int* count_out) {
    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Error: no se pudo abrir calibracion global '%s'\n", path);
        return -1;
    }

    char line[1024];
    int count = 0;
    size_t line_no = 0;
    while (fgets(line, sizeof(line), f)) {
        line_no++;
        trim_right(line);
        if (line[0] == '\0' || line[0] == '#') continue;

        char* tmp = line;
        while (*tmp == ' ' || *tmp == '\t') tmp++;
        if ((*tmp < '0' || *tmp > '9') && *tmp != '-') continue;

        char* tok[8] = {0};
        int ntok = 0;
        char* saveptr = NULL;
        char* part = strtok_r(tmp, ",\t ", &saveptr);
        while (part && ntok < 8) {
            tok[ntok++] = part;
            part = strtok_r(NULL, ",\t ", &saveptr);
        }
        if (ntok < 3) {
            fprintf(stderr, "Error: linea de calibracion global invalida %zu\n", line_no);
            fclose(f);
            return -1;
        }
        if (count >= MAX_POINTS) {
            fprintf(stderr, "Error: demasiados puntos de calibracion global\n");
            fclose(f);
            return -1;
        }

        CalibPoint p;
        if (parse_int(tok[0], &p.q) != 0 || parse_double(tok[1], &p.psnr_db) != 0 ||
            parse_double(tok[2], &p.mse) != 0) {
            fprintf(stderr, "Error: linea de calibracion global invalida %zu\n", line_no);
            fclose(f);
            return -1;
        }
        if (p.q < 0 || p.q > 255 || p.mse <= 0.0 || p.psnr_db <= 0.0) {
            fprintf(stderr, "Error: valores fuera de rango en linea %zu\n", line_no);
            fclose(f);
            return -1;
        }
        points[count++] = p;
    }
    fclose(f);

    if (count < 2) {
        fprintf(stderr, "Error: se requieren al menos dos puntos de calibracion global\n");
        return -1;
    }
    qsort(points, (size_t)count, sizeof(points[0]), cmp_q);
    *count_out = count;
    return 0;
}

static double psnr_to_mse(double psnr_db) {
    return (MAX_U16 * MAX_U16) / pow(10.0, psnr_db / 10.0);
}

static double mse_to_psnr(double mse) {
    return 10.0 * log10((MAX_U16 * MAX_U16) / mse);
}

static int select_q_from_psnr(
    const CalibPoint* points,
    int count,
    double target_psnr,
    double* q_float_out,
    double* predicted_psnr_out,
    double* predicted_mse_out,
    const char** status,
    const char** reason
) {
    int min_i = 0;
    int max_i = 0;
    for (int i = 1; i < count; ++i) {
        if (points[i].psnr_db < points[min_i].psnr_db) min_i = i;
        if (points[i].psnr_db > points[max_i].psnr_db) max_i = i;
    }
    if (target_psnr <= points[min_i].psnr_db) {
        *q_float_out = (double)points[min_i].q;
        *predicted_psnr_out = points[min_i].psnr_db;
        *predicted_mse_out = points[min_i].mse;
        *status = "clamped_low";
        *reason = "target_below_measured_range";
        return points[min_i].q;
    }
    if (target_psnr >= points[max_i].psnr_db) {
        *q_float_out = (double)points[max_i].q;
        *predicted_psnr_out = points[max_i].psnr_db;
        *predicted_mse_out = points[max_i].mse;
        *status = "clamped_high";
        *reason = "target_above_measured_range";
        return points[max_i].q;
    }

    for (int i = 0; i < count - 1; ++i) {
        double p0 = points[i].psnr_db;
        double p1 = points[i + 1].psnr_db;
        if ((target_psnr >= p0 && target_psnr <= p1) || (target_psnr >= p1 && target_psnr <= p0)) {
            double denom = p1 - p0;
            double t = (fabs(denom) < 1e-12) ? 0.0 : ((target_psnr - p0) / denom);
            double qf = (double)points[i].q + t * (double)(points[i + 1].q - points[i].q);
            *q_float_out = qf;
            *predicted_psnr_out = target_psnr;
            *predicted_mse_out = psnr_to_mse(target_psnr);
            *status = "reachable";
            *reason = "linear_interpolation_psnr";
            return clamp_int((int)lround(qf), 0, 255);
        }
    }

    *q_float_out = (double)points[max_i].q;
    *predicted_psnr_out = points[max_i].psnr_db;
    *predicted_mse_out = points[max_i].mse;
    *status = "clamped_high";
    *reason = "non_monotonic_fallback";
    return points[max_i].q;
}

static int select_q_from_mse(
    const CalibPoint* points,
    int count,
    double target_mse,
    double* q_float_out,
    double* predicted_psnr_out,
    double* predicted_mse_out,
    const char** status,
    const char** reason
) {
    int worst_i = 0;
    int best_i = 0;
    for (int i = 1; i < count; ++i) {
        if (points[i].mse > points[worst_i].mse) worst_i = i;
        if (points[i].mse < points[best_i].mse) best_i = i;
    }
    if (target_mse >= points[worst_i].mse) {
        *q_float_out = (double)points[worst_i].q;
        *predicted_mse_out = points[worst_i].mse;
        *predicted_psnr_out = points[worst_i].psnr_db;
        *status = "clamped_low";
        *reason = "target_mse_above_measured_range";
        return points[worst_i].q;
    }
    if (target_mse <= points[best_i].mse) {
        *q_float_out = (double)points[best_i].q;
        *predicted_mse_out = points[best_i].mse;
        *predicted_psnr_out = points[best_i].psnr_db;
        *status = "clamped_high";
        *reason = "target_mse_below_measured_range";
        return points[best_i].q;
    }

    for (int i = 0; i < count - 1; ++i) {
        double m0 = points[i].mse;
        double m1 = points[i + 1].mse;
        if ((target_mse <= m0 && target_mse >= m1) || (target_mse <= m1 && target_mse >= m0)) {
            double denom = m1 - m0;
            double t = (fabs(denom) < 1e-12) ? 0.0 : ((target_mse - m0) / denom);
            double qf = (double)points[i].q + t * (double)(points[i + 1].q - points[i].q);
            *q_float_out = qf;
            *predicted_mse_out = target_mse;
            *predicted_psnr_out = mse_to_psnr(target_mse);
            *status = "reachable";
            *reason = "linear_interpolation_mse";
            return clamp_int((int)lround(qf), 0, 255);
        }
    }

    *q_float_out = (double)points[best_i].q;
    *predicted_mse_out = points[best_i].mse;
    *predicted_psnr_out = points[best_i].psnr_db;
    *status = "clamped_high";
    *reason = "non_monotonic_fallback";
    return points[best_i].q;
}

static int write_qmap(const char* path, int q, size_t count) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "Error: no se pudo escribir Q-map '%s'\n", path);
        return -1;
    }
    uint8_t value = (uint8_t)q;
    for (size_t i = 0; i < count; ++i) {
        if (fwrite(&value, 1, 1, f) != 1) {
            fprintf(stderr, "Error: escritura incompleta de Q-map\n");
            fclose(f);
            return -1;
        }
    }
    fclose(f);
    return 0;
}

static int write_summary(
    const Options* opt,
    int selected_q,
    double q_float,
    double predicted_psnr,
    double predicted_mse,
    const char* status,
    const char* reason
) {
    if (!opt->summary_tsv_path) return 0;
    FILE* f = fopen(opt->summary_tsv_path, "w");
    if (!f) {
        fprintf(stderr, "Error: no se pudo escribir summary TSV '%s'\n", opt->summary_tsv_path);
        return -1;
    }
    double lambda = ((double)selected_q / 255.0) * opt->max_lambda;
    const char* mode = (opt->mode == TARGET_PSNR) ? "target_psnr" : "target_mse";
    fprintf(f, "mode\ttarget_value\tselected_q\tq_float\tlambda_equivalent\tpredicted_psnr_db\tpredicted_mse\tstatus\treason\twidth\theight\n");
    fprintf(f, "%s\t%.12g\t%d\t%.12g\t%.12g\t%.12g\t%.12g\t%s\t%s\t%d\t%d\n",
            mode, opt->target_value, selected_q, q_float, lambda,
            predicted_psnr, predicted_mse, status, reason, opt->width, opt->height);
    fclose(f);
    return 0;
}

int main(int argc, char** argv) {
    Options opt;
    if (parse_args(argc, argv, &opt) != 0) {
        usage(argv[0]);
        return 2;
    }

    CalibPoint points[MAX_POINTS];
    int count = 0;
    if (load_calibration(opt.calibration_path, points, &count) != 0) {
        return 1;
    }

    double q_float = NAN;
    double predicted_psnr = NAN;
    double predicted_mse = NAN;
    const char* status = "unknown";
    const char* reason = "unknown";
    int selected_q = 0;
    if (opt.mode == TARGET_PSNR) {
        selected_q = select_q_from_psnr(points, count, opt.target_value, &q_float,
                                        &predicted_psnr, &predicted_mse, &status, &reason);
    } else {
        selected_q = select_q_from_mse(points, count, opt.target_value, &q_float,
                                      &predicted_psnr, &predicted_mse, &status, &reason);
    }

    size_t qmap_count = (size_t)opt.width * (size_t)opt.height;
    if (write_qmap(opt.output_qmap_path, selected_q, qmap_count) != 0) {
        return 1;
    }
    if (write_summary(&opt, selected_q, q_float, predicted_psnr, predicted_mse, status, reason) != 0) {
        return 1;
    }

    double lambda = ((double)selected_q / 255.0) * opt.max_lambda;
    printf("Global target qmap: target=%s %.6f selected_q=%d lambda=%.9g status=%s reason=%s\n",
           opt.mode == TARGET_PSNR ? "psnr" : "mse",
           opt.target_value,
           selected_q,
           lambda,
           status,
           reason);
    printf("Predicted: PSNR=%.6f dB MSE=%.9g q_float=%.6f\n",
           predicted_psnr,
           predicted_mse,
           q_float);
    return 0;
}
