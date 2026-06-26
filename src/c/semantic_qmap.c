#define _POSIX_C_SOURCE 200112L
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_IMAGE_WIDTH 512
#define DEFAULT_IMAGE_HEIGHT 512
#define DEFAULT_BANDS 8
#define DEFAULT_BLOCK_SIZE 16
#define MAX_SAMPLE_VALUE 65535.0

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
    BAND_B01 = 0,
    BAND_B02,
    BAND_B03,
    BAND_B04,
    BAND_B05,
    BAND_B06,
    BAND_B07,
    BAND_B08,
    BAND_B8A,
    BAND_B09,
    BAND_B10,
    BAND_B11,
    BAND_B12,
    BAND_COUNT
} SentinelBand;

typedef struct {
    int index[BAND_COUNT];
    int loaded;
} BandMap;

typedef enum {
    LAYOUT_AUTO = 0,
    LAYOUT_SENTINEL2_8,
    LAYOUT_SENTINEL2_13,
    LAYOUT_BAND_MAP
} BandLayout;

typedef enum {
    PRESET_UNKNOWN = 0,
    PRESET_VEGETATION,
    PRESET_WATER,
    PRESET_BURNED,
    PRESET_SNOW,
    PRESET_WATER_BODY,
    PRESET_CHLOROPHYLL,
    PRESET_VEGETATION_GREEN,
    PRESET_CLOUDS,
    PRESET_BARREN_SOIL,
    PRESET_BURNED_AREA,
    PRESET_DARK_REGIONS,
    PRESET_LOCAL_CONTRAST,
    PRESET_LOW_NDVI,
    PRESET_HIGH_NDVI,
    PRESET_CLOUD_AVOID,
    PRESET_UNIFORM,
    PRESET_MANUAL
} SemanticPreset;

typedef enum {
    EVAL_NORMALIZED_DIFF = 0,
    EVAL_CLOUD_CBY,
    EVAL_VISIBLE_BRIGHTNESS,
    EVAL_VISIBLE_CONTRAST,
    EVAL_BSI,
    EVAL_BAIS2
} EvalMode;

typedef enum {
    CMP_GE = 0,
    CMP_LE
} CompareMode;

typedef enum {
    POLICY_BOOST_ONLY = 0,
    POLICY_FOCUS,
    POLICY_TARGET_FOCUS,
    POLICY_PRESERVE_ROI
} SemanticPolicy;

typedef struct {
    SemanticPreset preset;
    const char* name;
    const char* index_name;
    EvalMode eval_mode;
    SentinelBand a;
    SentinelBand b;
    double threshold;
    CompareMode cmp;
} PresetSpec;

typedef struct {
    const char* input_path;
    const char* calibration_path;
    const char* output_qmap_path;
    const char* summary_tsv_path;
    const char* band_map_path;
    const char* roi_map_path;
    const char* roi_tsv_path;
    const char* roi_command_path;
    SemanticPreset preset;
    BandLayout layout;
    int bands;
    int image_height;
    int image_width;
    int block_size;
    int q_mean;
    double adaptive_strength;
    int semantic_boost;
    int foreground_boost;
    int foreground_q;
    int foreground_q_set;
    int background_penalty;
    int background_q;
    int background_q_set;
    int allow_experimental_low_q;
    SemanticPolicy policy;
    int threshold_set;
    double threshold;
    int roi_target_psnr_set;
    double roi_target_psnr;
    int roi_target_mse_set;
    double roi_target_mse;
} Options;

typedef struct {
    int semantic_possible;
    int semantic_matches;
    int boost_applied;
    int boost_clamped;
    int penalty_applied;
    int penalty_clamped;
    int fixed_q_applied;
    int fixed_q_clamped;
    int target_applied;
    int target_clamped;
    int missing_bands;
    int no_valid_pixels;
} SemanticSummary;

static const char* BAND_NAMES[BAND_COUNT] = {
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B10", "B11", "B12"
};

static const PresetSpec PRESET_SPECS[] = {
    /* Tipo 1: indice normalizado (A-B)/(A+B) */
    {PRESET_VEGETATION,       "vegetation",       "NDVI",  EVAL_NORMALIZED_DIFF, BAND_B08, BAND_B04, 0.4, CMP_GE},
    {PRESET_WATER,            "water",            "NDMI",  EVAL_NORMALIZED_DIFF, BAND_B08, BAND_B11, 0.2, CMP_GE},
    {PRESET_BURNED,           "burned",           "NBR",   EVAL_NORMALIZED_DIFF, BAND_B08, BAND_B12, 0.1, CMP_LE},
    {PRESET_SNOW,             "snow",             "NDSI",  EVAL_NORMALIZED_DIFF, BAND_B03, BAND_B11, 0.4, CMP_GE},
    {PRESET_WATER_BODY,       "water_body",       "NDWI",  EVAL_NORMALIZED_DIFF, BAND_B03, BAND_B08, 0.1, CMP_GE},
    {PRESET_CHLOROPHYLL,      "chlorophyll",      "NDCI",  EVAL_NORMALIZED_DIFF, BAND_B05, BAND_B04, 0.1, CMP_GE},
    {PRESET_VEGETATION_GREEN, "vegetation_green", "GNDVI", EVAL_NORMALIZED_DIFF, BAND_B08, BAND_B03, 0.5, CMP_GE},
    /* Tipo 2: evaluaciones multi-banda */
    {PRESET_CLOUDS,           "clouds",           "CBY",   EVAL_CLOUD_CBY,       BAND_B03, BAND_B04, 0.5, CMP_GE},
    {PRESET_DARK_REGIONS,     "dark_regions",     "VIS_MEAN", EVAL_VISIBLE_BRIGHTNESS, BAND_B02, BAND_B03, 0.26, CMP_LE},
    {PRESET_LOCAL_CONTRAST,   "local_contrast",   "VIS_STD",  EVAL_VISIBLE_CONTRAST,   BAND_B02, BAND_B03, 0.035, CMP_GE},
    {PRESET_LOW_NDVI,         "low_ndvi",         "NDVI",     EVAL_NORMALIZED_DIFF,    BAND_B08, BAND_B04, 0.15, CMP_LE},
    {PRESET_HIGH_NDVI,        "high_ndvi",        "NDVI",     EVAL_NORMALIZED_DIFF,    BAND_B08, BAND_B04, 0.50, CMP_GE},
    {PRESET_CLOUD_AVOID,      "cloud_avoid",      "CBY_CLEAR", EVAL_CLOUD_CBY,         BAND_B03, BAND_B04, 0.5, CMP_LE},
    {PRESET_BARREN_SOIL,      "barren_soil",      "BSI",   EVAL_BSI,             BAND_B04, BAND_B08, 0.0, CMP_GE},
    {PRESET_BURNED_AREA,      "burned_area",      "BAIS2", EVAL_BAIS2,           BAND_B04, BAND_B12, 0.5, CMP_GE},
    /* Especiales */
    {PRESET_UNIFORM,          "uniform",          "none",       EVAL_NORMALIZED_DIFF, BAND_B01, BAND_B01, 0.0, CMP_GE},
    {PRESET_MANUAL,           "manual",           "manual_roi", EVAL_NORMALIZED_DIFF, BAND_B01, BAND_B01, 0.5, CMP_GE},
};

static void usage(const char* argv0) {
    fprintf(stderr,
        "Uso: %s --input image.raw --calibration fq_calibration.tsv --preset vegetation "
        "--output-qmap qmap.bin [opciones]\n\n"
        "Opciones:\n"
        "  --bands N                 Bandas BSQ uint16 (default: 8)\n"
        "  --height N                Alto de imagen (default: 512)\n"
        "  --width N                 Ancho de imagen (default: 512)\n"
        "  --block-size N            Tamano de bloque/Q-map (default: 16)\n"
        "  --band-layout L           auto|sentinel2-8|sentinel2-13 (default: auto)\n"
        "  --band-map path           TSV band_name<TAB>index para layouts personalizados\n"
        "  --roi-map path            ROI manual uint8 a resolucion Q-map; !=0 es ROI\n"
        "  --roi-tsv path            ROI manual TSV con columnas block_y,block_x\n"
        "  --roi-command path        ROI manual compacta SROI1 con CRC32 obligatorio\n"
        "  --summary-tsv path        Guarda resumen semantico por bloque\n"
        "  --q-mean Q                Presupuesto medio adaptativo (default: 204)\n"
        "  --adaptive-strength S     Intensidad adaptativa (default: 8)\n"
        "  --semantic-policy P       boost-only|focus|target-focus|preserve-roi (default: boost-only)\n"
        "  --foreground-boost Q      Incremento Q en ROI semantica (default: 8)\n"
        "  --foreground-q Q          Q fijo en ROI para preserve-roi (default: 255)\n"
        "  --background-penalty Q    Reduccion Q fuera de ROI en focus (default: 0)\n"
        "  --background-q Q          Q fijo fuera de ROI en focus/preserve-roi; precede a penalty\n"
        "  --allow-experimental-low-q Permite background-q experimental 64..127\n"
        "  --roi-target-psnr X       PSNR objetivo en ROI para target-focus\n"
        "  --roi-target-mse X        MSE objetivo en ROI para target-focus\n"
        "  --semantic-boost Q        Alias legacy de --foreground-boost\n"
        "  --threshold X             Umbral manual del indice\n\n"
        "Presets disponibles:\n"
        "  Tipo 1 (indice normalizado):\n"
        "    vegetation(NDVI B08-B04)  water(NDMI B08-B11)  burned(NBR B08-B12)\n"
        "    snow(NDSI B03-B11)  water_body(NDWI B03-B08 >= 0.10)  chlorophyll(NDCI B05-B04)\n"
        "    vegetation_green(GNDVI B08-B03)\n"
        "  Tipo 2 (multi-banda):\n"
        "    clouds(CBY B03,B04[,B11])  barren_soil(BSI B02,B04,B08,B11)\n"
        "    burned_area(BAIS2 B04,B06,B07,B8A,B12)\n"
        "  Operativos automaticos 8 bandas:\n"
        "    dark_regions(VIS_MEAN B02,B03,B04 <= 0.26)\n"
        "    local_contrast(VIS_STD B02,B03,B04 >= 0.035)\n"
        "    low_ndvi(NDVI B08,B04 <= 0.15)  high_ndvi(NDVI B08,B04 >= 0.50)\n"
        "    cloud_avoid(CBY B03,B04[,B11] <= 0.50; protege no-nube)\n"
        "  Especiales:\n"
        "    uniform  manual\n\n"
        "La imagen canonica de 8 bandas se interpreta como B02,B03,B04,B05,B06,B07,B08,B8A.\n",
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

static SemanticPreset parse_preset(const char* s) {
    for (size_t i = 0; i < sizeof(PRESET_SPECS) / sizeof(PRESET_SPECS[0]); ++i) {
        if (strcmp(s, PRESET_SPECS[i].name) == 0) {
            return PRESET_SPECS[i].preset;
        }
    }
    return PRESET_UNKNOWN;
}

static SentinelBand parse_band_name(const char* s) {
    for (int i = 0; i < BAND_COUNT; ++i) {
        if (strcmp(s, BAND_NAMES[i]) == 0) return (SentinelBand)i;
    }
    return BAND_COUNT;
}

static BandLayout parse_layout(const char* s) {
    if (strcmp(s, "auto") == 0) return LAYOUT_AUTO;
    if (strcmp(s, "sentinel2-8") == 0) return LAYOUT_SENTINEL2_8;
    if (strcmp(s, "sentinel2-13") == 0) return LAYOUT_SENTINEL2_13;
    return LAYOUT_AUTO;
}

static SemanticPolicy parse_policy(const char* s, int* ok) {
    *ok = 1;
    if (strcmp(s, "boost-only") == 0) return POLICY_BOOST_ONLY;
    if (strcmp(s, "focus") == 0) return POLICY_FOCUS;
    if (strcmp(s, "target-focus") == 0) return POLICY_TARGET_FOCUS;
    if (strcmp(s, "preserve-roi") == 0) return POLICY_PRESERVE_ROI;
    *ok = 0;
    return POLICY_BOOST_ONLY;
}

static const char* policy_name(SemanticPolicy policy) {
    switch (policy) {
        case POLICY_FOCUS: return "focus";
        case POLICY_TARGET_FOCUS: return "target-focus";
        case POLICY_PRESERVE_ROI: return "preserve-roi";
        default: return "boost-only";
    }
}

static int policy_uses_focus_background(SemanticPolicy policy) {
    return policy == POLICY_FOCUS || policy == POLICY_TARGET_FOCUS || policy == POLICY_PRESERVE_ROI;
}

static const char* layout_name(BandLayout layout) {
    switch (layout) {
        case LAYOUT_SENTINEL2_8: return "sentinel2-8";
        case LAYOUT_SENTINEL2_13: return "sentinel2-13";
        case LAYOUT_BAND_MAP: return "band-map";
        default: return "auto";
    }
}

static const PresetSpec* spec_for_preset(SemanticPreset preset) {
    for (size_t i = 0; i < sizeof(PRESET_SPECS) / sizeof(PRESET_SPECS[0]); ++i) {
        if (PRESET_SPECS[i].preset == preset) return &PRESET_SPECS[i];
    }
    return NULL;
}

static int parse_args(int argc, char** argv, Options* opt) {
    memset(opt, 0, sizeof(*opt));
    opt->bands = DEFAULT_BANDS;
    opt->image_height = DEFAULT_IMAGE_HEIGHT;
    opt->image_width = DEFAULT_IMAGE_WIDTH;
    opt->block_size = DEFAULT_BLOCK_SIZE;
    opt->q_mean = 204;
    opt->adaptive_strength = 8.0;
    opt->semantic_boost = 8;
    opt->foreground_boost = 8;
    opt->foreground_q = 255;
    opt->foreground_q_set = 0;
    opt->background_penalty = 0;
    opt->background_q = 0;
    opt->background_q_set = 0;
    opt->allow_experimental_low_q = 0;
    opt->policy = POLICY_BOOST_ONLY;
    opt->layout = LAYOUT_AUTO;

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (strcmp(a, "--input") == 0 && i + 1 < argc) {
            opt->input_path = argv[++i];
        } else if (strcmp(a, "--calibration") == 0 && i + 1 < argc) {
            opt->calibration_path = argv[++i];
        } else if (strcmp(a, "--output-qmap") == 0 && i + 1 < argc) {
            opt->output_qmap_path = argv[++i];
        } else if (strcmp(a, "--summary-tsv") == 0 && i + 1 < argc) {
            opt->summary_tsv_path = argv[++i];
        } else if (strcmp(a, "--band-map") == 0 && i + 1 < argc) {
            opt->band_map_path = argv[++i];
            opt->layout = LAYOUT_BAND_MAP;
        } else if (strcmp(a, "--roi-map") == 0 && i + 1 < argc) {
            opt->roi_map_path = argv[++i];
        } else if (strcmp(a, "--roi-tsv") == 0 && i + 1 < argc) {
            opt->roi_tsv_path = argv[++i];
        } else if (strcmp(a, "--roi-command") == 0 && i + 1 < argc) {
            opt->roi_command_path = argv[++i];
        } else if (strcmp(a, "--preset") == 0 && i + 1 < argc) {
            opt->preset = parse_preset(argv[++i]);
            if (opt->preset == PRESET_UNKNOWN) {
                fprintf(stderr, "Error: preset desconocido.\n");
                return -1;
            }
        } else if (strcmp(a, "--band-layout") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            opt->layout = parse_layout(value);
            if (strcmp(value, layout_name(opt->layout)) != 0 && strcmp(value, "auto") != 0) {
                fprintf(stderr, "Error: --band-layout debe ser auto, sentinel2-8 o sentinel2-13.\n");
                return -1;
            }
        } else if (strcmp(a, "--bands") == 0 && i + 1 < argc) {
            if (parse_int(argv[++i], &opt->bands) != 0 || opt->bands <= 0) return -1;
        } else if (strcmp(a, "--height") == 0 && i + 1 < argc) {
            if (parse_int(argv[++i], &opt->image_height) != 0 || opt->image_height <= 0) return -1;
        } else if (strcmp(a, "--width") == 0 && i + 1 < argc) {
            if (parse_int(argv[++i], &opt->image_width) != 0 || opt->image_width <= 0) return -1;
        } else if (strcmp(a, "--block-size") == 0 && i + 1 < argc) {
            if (parse_int(argv[++i], &opt->block_size) != 0 || opt->block_size <= 0) return -1;
        } else if (strcmp(a, "--q-mean") == 0 && i + 1 < argc) {
            if (parse_int(argv[++i], &opt->q_mean) != 0 || opt->q_mean < 0 || opt->q_mean > 255) return -1;
        } else if (strcmp(a, "--adaptive-strength") == 0 && i + 1 < argc) {
            if (parse_double(argv[++i], &opt->adaptive_strength) != 0 || opt->adaptive_strength < 0.0) return -1;
        } else if (strcmp(a, "--semantic-policy") == 0 && i + 1 < argc) {
            int ok = 0;
            opt->policy = parse_policy(argv[++i], &ok);
            if (!ok) {
                fprintf(stderr, "Error: --semantic-policy debe ser boost-only, focus, target-focus o preserve-roi.\n");
                return -1;
            }
        } else if (strcmp(a, "--foreground-boost") == 0 && i + 1 < argc) {
            if (parse_int(argv[++i], &opt->foreground_boost) != 0 ||
                opt->foreground_boost < 0 || opt->foreground_boost > 255) {
                fprintf(stderr, "Error: --foreground-boost debe estar en [0,255].\n");
                return -1;
            }
        } else if (strcmp(a, "--semantic-boost") == 0 && i + 1 < argc) {
            if (parse_int(argv[++i], &opt->semantic_boost) != 0 || opt->semantic_boost < 0 || opt->semantic_boost > 255) return -1;
            opt->foreground_boost = opt->semantic_boost;
        } else if (strcmp(a, "--foreground-q") == 0 && i + 1 < argc) {
            if (parse_int(argv[++i], &opt->foreground_q) != 0 ||
                opt->foreground_q < 0 || opt->foreground_q > 255) {
                fprintf(stderr, "Error: --foreground-q debe estar en [0,255].\n");
                return -1;
            }
            opt->foreground_q_set = 1;
        } else if (strcmp(a, "--background-penalty") == 0 && i + 1 < argc) {
            if (parse_int(argv[++i], &opt->background_penalty) != 0 ||
                opt->background_penalty < 0 || opt->background_penalty > 255) {
                fprintf(stderr, "Error: --background-penalty debe estar en [0,255].\n");
                return -1;
            }
        } else if (strcmp(a, "--background-q") == 0 && i + 1 < argc) {
            if (parse_int(argv[++i], &opt->background_q) != 0 ||
                opt->background_q < 0 || opt->background_q > 255) {
                fprintf(stderr, "Error: --background-q debe estar en [0,255].\n");
                return -1;
            }
            opt->background_q_set = 1;
        } else if (strcmp(a, "--allow-experimental-low-q") == 0) {
            opt->allow_experimental_low_q = 1;
        } else if (strcmp(a, "--roi-target-psnr") == 0 && i + 1 < argc) {
            if (parse_double(argv[++i], &opt->roi_target_psnr) != 0 || opt->roi_target_psnr <= 0.0) {
                fprintf(stderr, "Error: --roi-target-psnr debe ser positivo.\n");
                return -1;
            }
            opt->roi_target_psnr_set = 1;
        } else if (strcmp(a, "--roi-target-mse") == 0 && i + 1 < argc) {
            if (parse_double(argv[++i], &opt->roi_target_mse) != 0 || opt->roi_target_mse <= 0.0) {
                fprintf(stderr, "Error: --roi-target-mse debe ser positivo.\n");
                return -1;
            }
            opt->roi_target_mse_set = 1;
        } else if (strcmp(a, "--threshold") == 0 && i + 1 < argc) {
            if (parse_double(argv[++i], &opt->threshold) != 0) return -1;
            opt->threshold_set = 1;
        } else if (strcmp(a, "-h") == 0 || strcmp(a, "--help") == 0) {
            usage(argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "Error: argumento desconocido o incompleto: %s\n", a);
            return -1;
        }
    }

    if (!opt->calibration_path || !opt->output_qmap_path || opt->preset == PRESET_UNKNOWN) {
        return -1;
    }
    if (opt->preset != PRESET_MANUAL && !opt->input_path) {
        fprintf(stderr, "Error: --input es obligatorio salvo con --preset manual.\n");
        return -1;
    }
    if (opt->preset == PRESET_MANUAL && !opt->roi_map_path && !opt->roi_tsv_path && !opt->roi_command_path) {
        fprintf(stderr, "Error: --preset manual requiere --roi-map, --roi-tsv o --roi-command.\n");
        return -1;
    }
    int roi_sources = 0;
    if (opt->roi_map_path) roi_sources++;
    if (opt->roi_tsv_path) roi_sources++;
    if (opt->roi_command_path) roi_sources++;
    if (roi_sources > 1) {
        fprintf(stderr, "Error: usa solo una fuente ROI: --roi-map, --roi-tsv o --roi-command.\n");
        return -1;
    }
    int target_sources = opt->roi_target_psnr_set + opt->roi_target_mse_set;
    if (opt->policy == POLICY_TARGET_FOCUS && target_sources != 1) {
        fprintf(stderr, "Error: target-focus requiere exactamente uno de --roi-target-psnr o --roi-target-mse.\n");
        return -1;
    }
    if (opt->policy != POLICY_TARGET_FOCUS && target_sources != 0) {
        fprintf(stderr, "Error: --roi-target-psnr/--roi-target-mse solo se usan con --semantic-policy target-focus.\n");
        return -1;
    }
    if (opt->policy == POLICY_PRESERVE_ROI && !opt->background_q_set) {
        fprintf(stderr, "Error: preserve-roi requiere --background-q para fijar la degradacion de no-ROI.\n");
        return -1;
    }
    if (opt->policy == POLICY_PRESERVE_ROI && opt->foreground_q < 128 && !opt->allow_experimental_low_q) {
        fprintf(stderr, "Error: --foreground-q < 128 requiere --allow-experimental-low-q.\n");
        return -1;
    }
    if (opt->policy == POLICY_PRESERVE_ROI && opt->foreground_q < 64) {
        fprintf(stderr, "Error: la ruta experimental solo permite foreground-q >= 64.\n");
        return -1;
    }
    if (opt->background_q_set && opt->background_q < 128 && !opt->allow_experimental_low_q) {
        fprintf(stderr, "Error: --background-q < 128 requiere --allow-experimental-low-q.\n");
        return -1;
    }
    if (opt->image_height % opt->block_size != 0 || opt->image_width % opt->block_size != 0) {
        fprintf(stderr, "Error: dimensiones no divisibles por --block-size.\n");
        return -1;
    }
    if (opt->allow_experimental_low_q) {
        if (!opt->background_q_set || opt->background_q >= 128) {
            fprintf(stderr, "Error: --allow-experimental-low-q requiere --background-q en [64,127].\n");
            return -1;
        }
        if (opt->background_q < 64) {
            fprintf(stderr, "Error: la ruta operativa experimental solo permite background-q >= 64.\n");
            return -1;
        }
    }
    return 0;
}

static BandLayout resolve_layout(BandLayout layout, int bands) {
    if (layout != LAYOUT_AUTO) return layout;
    if (bands == 13) return LAYOUT_SENTINEL2_13;
    return LAYOUT_SENTINEL2_8;
}

static int band_index(SentinelBand band, BandLayout layout, int bands, const BandMap* band_map) {
    static const int sentinel2_8[BAND_COUNT] = {
        -1, 0, 1, 2, 3, 4, 5, 6, 7, -1, -1, -1, -1
    };
    int idx = -1;
    if (layout == LAYOUT_BAND_MAP) {
        idx = band_map ? band_map->index[(int)band] : -1;
    } else if (layout == LAYOUT_SENTINEL2_13) {
        idx = (int)band;
    } else if (layout == LAYOUT_SENTINEL2_8) {
        idx = sentinel2_8[(int)band];
    }
    if (idx < 0 || idx >= bands) return -1;
    return idx;
}

static void init_band_map(BandMap* map) {
    for (int i = 0; i < BAND_COUNT; ++i) map->index[i] = -1;
    map->loaded = 0;
}

static int load_band_map(const char* path, int bands, BandMap* map) {
    init_band_map(map);
    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Error: no se pudo abrir band-map '%s'\n", path);
        return -1;
    }

    char line[512];
    size_t line_no = 0;
    while (fgets(line, sizeof(line), f)) {
        line_no++;
        if (line[0] == '#' || line[0] == '\n' || strncmp(line, "band_name", 9) == 0) {
            continue;
        }

        char band_name[32];
        int idx = -1;
        int n = sscanf(line, "%31s\t%d", band_name, &idx);
        if (n != 2) n = sscanf(line, "%31s %d", band_name, &idx);
        if (n != 2) {
            fprintf(stderr, "Error: linea band-map invalida %zu\n", line_no);
            fclose(f);
            return -1;
        }

        SentinelBand band = parse_band_name(band_name);
        if (band == BAND_COUNT) {
            fprintf(stderr, "Error: banda desconocida '%s' en linea %zu\n", band_name, line_no);
            fclose(f);
            return -1;
        }
        if (idx < 0 || idx >= bands) {
            fprintf(stderr, "Error: indice fuera de rango en band-map linea %zu\n", line_no);
            fclose(f);
            return -1;
        }
        map->index[(int)band] = idx;
        map->loaded = 1;
    }
    fclose(f);
    return 0;
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
    double log_std
) {
    if (!c->valid || c->mse_at_baseline <= 0.0 || log_std <= 0.0) {
        return clamp_int(c->q_baseline, c->q_min, c->q_max);
    }
    double z = (log(c->mse_at_baseline) - log_mean) / log_std;
    double q_float = (double)q_mean + (strength * z);
    int q_raw = (int)lround(q_float);
    return clamp_int(q_raw, c->q_min, c->q_max);
}

static double target_mse_from_psnr(double psnr_db) {
    return (MAX_SAMPLE_VALUE * MAX_SAMPLE_VALUE) / pow(10.0, psnr_db / 10.0);
}

static double lambda_from_q_model(int q, double max_lambda) {
    return ((double)q / 255.0) * max_lambda;
}

static double mod_from_lambda(const FQBlockCalib* c, double lambda) {
    return c->mod_a * lambda + c->mod_b;
}

static double predicted_mse_for_q(const FQBlockCalib* c, int q) {
    double lambda = lambda_from_q_model(q, c->max_lambda);
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
    if (target_mse < best_mse) return "too_strict";
    if (target_mse > worst_mse) return "too_relaxed";
    return "reachable";
}

static int select_q_for_target_mse(const FQBlockCalib* c, double target_mse, const char** reason, const char** feasibility) {
    *feasibility = feasibility_for_target(c, target_mse);
    if (!c->valid || c->c1 <= 0.0 || c->mod_a <= 0.0 || c->max_lambda <= 0.0) {
        *reason = "target_fallback_invalid_calibration";
        return clamp_int(c->q_baseline, c->q_min, c->q_max);
    }
    if (!(target_mse > c->c0)) {
        *reason = "target_clamped_high_quality_floor";
        return c->q_max;
    }

    double target_mod = sqrt(c->c1 / (target_mse - c->c0));
    double lambda = (target_mod - c->mod_b) / c->mod_a;
    double q_float = (lambda / c->max_lambda) * 255.0;
    if (!isfinite(q_float)) {
        *reason = "target_fallback_nonfinite";
        return clamp_int(c->q_baseline, c->q_min, c->q_max);
    }

    int q_raw = (int)lround(q_float);
    int q = clamp_int(q_raw, c->q_min, c->q_max);
    if (q != q_raw) {
        *reason = (q_raw < c->q_min) ? "target_clamped_low_q" : "target_clamped_high_q";
    } else {
        *reason = "target_model";
    }
    return q;
}

static int select_background_fixed_q(const Options* opt, const FQBlockCalib* c, const char** range_status) {
    if (opt->allow_experimental_low_q && opt->background_q >= 64 && opt->background_q < 128) {
        *range_status = "experimental_low_q";
        return clamp_int(opt->background_q, 0, c->q_max);
    }
    int fixed_q = clamp_int(opt->background_q, c->q_min, c->q_max);
    if (opt->background_q < c->q_min) {
        *range_status = "official_clamped_low_q";
    } else {
        *range_status = "official";
    }
    return fixed_q;
}

static int select_foreground_fixed_q(const Options* opt, const FQBlockCalib* c, const char** range_status) {
    if (opt->allow_experimental_low_q && opt->foreground_q >= 64 && opt->foreground_q < 128) {
        *range_status = "experimental_low_q";
        return clamp_int(opt->foreground_q, 0, c->q_max);
    }
    int fixed_q = clamp_int(opt->foreground_q, c->q_min, c->q_max);
    if (opt->foreground_q < c->q_min) {
        *range_status = "official_clamped_low_q";
    } else if (opt->foreground_q > c->q_max) {
        *range_status = "official_clamped_high_q";
    } else {
        *range_status = "official";
    }
    return fixed_q;
}

static int load_calibration(const char* path, FQBlockCalib* blocks, int q_width, int q_height) {
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
        if (c.block_y < 0 || c.block_y >= q_height || c.block_x < 0 || c.block_x >= q_width) {
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
        blocks[c.block_y * q_width + c.block_x] = c;
        loaded++;
    }
    fclose(f);

    int expected = q_width * q_height;
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

static uint16_t* load_raw_u16(const char* path, int bands, int height, int width) {
    size_t pixels_per_band = (size_t)height * (size_t)width;
    size_t total = (size_t)bands * pixels_per_band;
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

static uint8_t* load_roi_map_u8(const char* path, int q_width, int q_height) {
    size_t total = (size_t)q_width * (size_t)q_height;
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error: no se pudo abrir ROI-map '%s'\n", path);
        return NULL;
    }
    uint8_t* roi = (uint8_t*)malloc(total);
    if (!roi) {
        fprintf(stderr, "Error: memoria insuficiente para ROI-map\n");
        fclose(f);
        return NULL;
    }
    size_t n = fread(roi, sizeof(uint8_t), total, f);
    int extra = fgetc(f);
    fclose(f);
    if (n != total || extra != EOF) {
        fprintf(stderr, "Error: ROI-map '%s' debe tener exactamente %zu bytes\n", path, total);
        free(roi);
        return NULL;
    }
    return roi;
}

static uint8_t* load_roi_tsv(const char* path, int q_width, int q_height) {
    size_t total = (size_t)q_width * (size_t)q_height;
    uint8_t* roi = (uint8_t*)calloc(total, sizeof(uint8_t));
    if (!roi) {
        fprintf(stderr, "Error: memoria insuficiente para ROI TSV\n");
        return NULL;
    }

    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Error: no se pudo abrir ROI TSV '%s'\n", path);
        free(roi);
        return NULL;
    }

    char line[512];
    size_t line_no = 0;
    while (fgets(line, sizeof(line), f)) {
        line_no++;
        if (line[0] == '#' || line[0] == '\n' || strncmp(line, "block_y", 7) == 0) {
            continue;
        }

        int by = -1;
        int bx = -1;
        int n = sscanf(line, "%d\t%d", &by, &bx);
        if (n != 2) n = sscanf(line, "%d,%d", &by, &bx);
        if (n != 2) n = sscanf(line, "%d %d", &by, &bx);
        if (n != 2) {
            fprintf(stderr, "Error: linea ROI TSV invalida %zu\n", line_no);
            fclose(f);
            free(roi);
            return NULL;
        }
        if (by < 0 || by >= q_height || bx < 0 || bx >= q_width) {
            fprintf(stderr, "Error: bloque ROI fuera de rango en linea %zu\n", line_no);
            fclose(f);
            free(roi);
            return NULL;
        }
        roi[by * q_width + bx] = 1;
    }
    fclose(f);
    return roi;
}

static uint32_t crc32_bytes(const unsigned char* data, size_t len) {
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; ++i) {
        crc ^= (uint32_t)data[i];
        for (int bit = 0; bit < 8; ++bit) {
            uint32_t mask = 0u - (crc & 1u);
            crc = (crc >> 1) ^ (0xEDB88320u & mask);
        }
    }
    return crc ^ 0xFFFFFFFFu;
}

static int hex_digit(int c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
    if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
    return -1;
}

static int parse_crc32_hex8(const char* s, uint32_t* out) {
    uint32_t value = 0u;
    for (int i = 0; i < 8; ++i) {
        int h = hex_digit((unsigned char)s[i]);
        if (h < 0) return -1;
        value = (value << 4) | (uint32_t)h;
    }
    if (s[8] != '\0') return -1;
    *out = value;
    return 0;
}

static void trim_right(char* s) {
    size_t n = strlen(s);
    while (n > 0) {
        unsigned char c = (unsigned char)s[n - 1];
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r') break;
        s[n - 1] = '\0';
        n--;
    }
}

static int parse_roi_range_token(const char* token, uint8_t* roi, int q_width, int q_height, int* blocks_set) {
    const char* colon = strchr(token, ':');
    if (!colon || colon == token || colon[1] == '\0') return -1;

    char row_buf[32];
    size_t row_len = (size_t)(colon - token);
    if (row_len >= sizeof(row_buf)) return -1;
    memcpy(row_buf, token, row_len);
    row_buf[row_len] = '\0';

    int row = -1;
    if (parse_int(row_buf, &row) != 0 || row < 0 || row >= q_height) return -1;

    const char* col_part = colon + 1;
    const char* dash = strchr(col_part, '-');
    int x0 = -1;
    int x1 = -1;
    if (dash) {
        if (dash == col_part || dash[1] == '\0') return -1;
        char x0_buf[32];
        char x1_buf[32];
        size_t x0_len = (size_t)(dash - col_part);
        size_t x1_len = strlen(dash + 1);
        if (x0_len >= sizeof(x0_buf) || x1_len >= sizeof(x1_buf)) return -1;
        memcpy(x0_buf, col_part, x0_len);
        x0_buf[x0_len] = '\0';
        memcpy(x1_buf, dash + 1, x1_len + 1);
        if (parse_int(x0_buf, &x0) != 0 || parse_int(x1_buf, &x1) != 0) return -1;
    } else {
        if (parse_int(col_part, &x0) != 0) return -1;
        x1 = x0;
    }
    if (x0 < 0 || x1 < 0 || x0 > x1 || x1 >= q_width) return -1;

    for (int x = x0; x <= x1; ++x) {
        size_t idx = (size_t)row * (size_t)q_width + (size_t)x;
        if (roi[idx] == 0) {
            roi[idx] = 1;
            (*blocks_set)++;
        }
    }
    return 0;
}

static uint8_t* load_roi_command(const char* path, int q_width, int q_height) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error: no se pudo abrir ROI command '%s'\n", path);
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    long file_size = ftell(f);
    if (file_size <= 0 || file_size > 65536) {
        fprintf(stderr, "Error: ROI command '%s' tiene tamano invalido\n", path);
        fclose(f);
        return NULL;
    }
    rewind(f);

    char* text = (char*)malloc((size_t)file_size + 1u);
    if (!text) {
        fprintf(stderr, "Error: memoria insuficiente para ROI command\n");
        fclose(f);
        return NULL;
    }
    size_t n = fread(text, 1, (size_t)file_size, f);
    fclose(f);
    if (n != (size_t)file_size) {
        fprintf(stderr, "Error: lectura incompleta de ROI command\n");
        free(text);
        return NULL;
    }
    text[n] = '\0';
    trim_right(text);

    const char* crc_marker = ";CRC32=";
    char* crc_pos = strstr(text, crc_marker);
    if (!crc_pos) {
        fprintf(stderr, "Error: ROI command requiere ;CRC32=XXXXXXXX\n");
        free(text);
        return NULL;
    }
    if (strstr(crc_pos + 1, crc_marker)) {
        fprintf(stderr, "Error: ROI command contiene multiples CRC32\n");
        free(text);
        return NULL;
    }

    uint32_t expected_crc = 0u;
    if (parse_crc32_hex8(crc_pos + strlen(crc_marker), &expected_crc) != 0) {
        fprintf(stderr, "Error: CRC32 invalido en ROI command\n");
        free(text);
        return NULL;
    }

    size_t payload_len = (size_t)(crc_pos - text);
    uint32_t actual_crc = crc32_bytes((const unsigned char*)text, payload_len);
    if (actual_crc != expected_crc) {
        fprintf(stderr, "Error: CRC32 ROI command no coincide: esperado %08X, calculado %08X\n",
                expected_crc, actual_crc);
        free(text);
        return NULL;
    }

    *crc_pos = '\0';
    char* payload = text;
    if (strncmp(payload, "SROI1;", 6) != 0) {
        fprintf(stderr, "Error: ROI command debe empezar por SROI1;\n");
        free(text);
        return NULL;
    }

    char* copy = (char*)malloc(strlen(payload) + 1u);
    if (!copy) {
        fprintf(stderr, "Error: memoria insuficiente para parsear ROI command\n");
        free(text);
        return NULL;
    }
    strcpy(copy, payload);

    uint8_t* roi = (uint8_t*)calloc((size_t)q_width * (size_t)q_height, sizeof(uint8_t));
    if (!roi) {
        fprintf(stderr, "Error: memoria insuficiente para ROI command\n");
        free(copy);
        free(text);
        return NULL;
    }

    int saw_grid = 0;
    int saw_roi = 0;
    int blocks_set = 0;
    char* saveptr = NULL;
    char* part = strtok_r(copy, ";", &saveptr);
    while (part) {
        if (strcmp(part, "SROI1") == 0) {
            /* version marker */
        } else if (strncmp(part, "GRID=", 5) == 0) {
            int gw = -1;
            int gh = -1;
            if (sscanf(part + 5, "%dx%d", &gw, &gh) != 2 || gw != q_width || gh != q_height) {
                fprintf(stderr, "Error: GRID de ROI command incompatible, esperado %dx%d\n", q_width, q_height);
                free(roi);
                free(copy);
                free(text);
                return NULL;
            }
            saw_grid = 1;
        } else if (strncmp(part, "ROI=", 4) == 0) {
            char* roi_part = part + 4;
            if (*roi_part == '\0') {
                fprintf(stderr, "Error: ROI command contiene ROI vacia\n");
                free(roi);
                free(copy);
                free(text);
                return NULL;
            }
            char* roi_save = NULL;
            char* token = strtok_r(roi_part, ",", &roi_save);
            while (token) {
                if (parse_roi_range_token(token, roi, q_width, q_height, &blocks_set) != 0) {
                    fprintf(stderr, "Error: rango ROI command invalido: '%s'\n", token);
                    free(roi);
                    free(copy);
                    free(text);
                    return NULL;
                }
                token = strtok_r(NULL, ",", &roi_save);
            }
            saw_roi = 1;
        } else if (*part != '\0') {
            fprintf(stderr, "Error: campo ROI command desconocido: '%s'\n", part);
            free(roi);
            free(copy);
            free(text);
            return NULL;
        }
        part = strtok_r(NULL, ";", &saveptr);
    }

    free(copy);
    free(text);
    if (!saw_grid || !saw_roi || blocks_set <= 0) {
        fprintf(stderr, "Error: ROI command requiere GRID, ROI no vacia y al menos un bloque\n");
        free(roi);
        return NULL;
    }
    return roi;
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

static int block_index_mean(
    const uint16_t* raw,
    int bands,
    int image_height,
    int image_width,
    int block_size,
    int block_y,
    int block_x,
    int band_a,
    int band_b,
    double* mean_out
) {
    size_t band_stride = (size_t)image_height * (size_t)image_width;
    int y0 = block_y * block_size;
    int x0 = block_x * block_size;
    double sum = 0.0;
    int valid = 0;
    (void)bands;

    for (int y = y0; y < y0 + block_size; ++y) {
        size_t row_offset = (size_t)y * (size_t)image_width;
        for (int x = x0; x < x0 + block_size; ++x) {
            size_t p = row_offset + (size_t)x;
            double va = (double)raw[(size_t)band_a * band_stride + p];
            double vb = (double)raw[(size_t)band_b * band_stride + p];
            double den = va + vb;
            if (fabs(den) < 1e-12) continue;
            sum += (va - vb) / den;
            valid++;
        }
    }

    if (valid <= 0) return -1;
    *mean_out = sum / (double)valid;
    return 0;
}

static int semantic_match(double index_mean, double threshold, CompareMode cmp) {
    if (cmp == CMP_LE) return index_mean <= threshold;
    return index_mean >= threshold;
}

/* ---------- Tipo 2a: Braaten-Cohen-Yang cloud detection ---------- */
/* cloud = (bRatio > 1) OR (bRatio > 0 AND NDGR > 0)               */
/* bRatio = (B03 - 0.175) / (0.39 - 0.175)                          */
/* NDGR = (B03 - B04) / (B03 + B04)                                 */
/* Si B11 disponible: cloud = cloud AND (B11 > 0.2)                  */
/* Devuelve fraccion de pixeles cloud en el bloque.                  */
static int block_cloud_cby_fraction(
    const uint16_t* raw,
    int bands,
    int image_height,
    int image_width,
    int block_size,
    int block_y,
    int block_x,
    int band_b03,
    int band_b04,
    int band_b11,
    double* fraction_out
) {
    /* B11 < 0 indica que no esta disponible (fallback a version basica) */
    size_t band_stride = (size_t)image_height * (size_t)image_width;
    int y0 = block_y * block_size;
    int x0 = block_x * block_size;
    int total = 0;
    int cloud_count = 0;
    (void)bands;

    /* Sentinel-2 L1C reflectancia: uint16 con factor 10000 => 1.0 = 10000 */
    const double SCALE = 1.0 / 10000.0;

    for (int y = y0; y < y0 + block_size; ++y) {
        size_t row_off = (size_t)y * (size_t)image_width;
        for (int x = x0; x < x0 + block_size; ++x) {
            size_t p = row_off + (size_t)x;
            double b03 = (double)raw[(size_t)band_b03 * band_stride + p] * SCALE;
            double b04 = (double)raw[(size_t)band_b04 * band_stride + p] * SCALE;

            double bRatio = (b03 - 0.175) / (0.39 - 0.175);
            double den = b03 + b04;
            double ndgr = (fabs(den) < 1e-12) ? 0.0 : (b03 - b04) / den;

            int is_cloud = (bRatio > 1.0) || (bRatio > 0.0 && ndgr > 0.0);

            /* Version mejorada con B11 (88% vs 73% precision) */
            if (is_cloud && band_b11 >= 0) {
                double b11 = (double)raw[(size_t)band_b11 * band_stride + p] * SCALE;
                if (b11 <= 0.2) is_cloud = 0;
            }

            if (is_cloud) cloud_count++;
            total++;
        }
    }

    if (total <= 0) return -1;
    *fraction_out = (double)cloud_count / (double)total;
    return 0;
}

/* ---------- Modos operativos por brillo visible ---------- */
/* brightness = mean((B02+B03+B04)/3) con reflectancia uint16/10000. */
static int block_visible_brightness_mean(
    const uint16_t* raw,
    int bands,
    int image_height,
    int image_width,
    int block_size,
    int block_y,
    int block_x,
    int band_b02,
    int band_b03,
    int band_b04,
    double* mean_out
) {
    size_t band_stride = (size_t)image_height * (size_t)image_width;
    int y0 = block_y * block_size;
    int x0 = block_x * block_size;
    double sum = 0.0;
    int total = 0;
    (void)bands;

    const double SCALE = 1.0 / 10000.0;

    for (int y = y0; y < y0 + block_size; ++y) {
        size_t row_off = (size_t)y * (size_t)image_width;
        for (int x = x0; x < x0 + block_size; ++x) {
            size_t p = row_off + (size_t)x;
            double b02 = (double)raw[(size_t)band_b02 * band_stride + p] * SCALE;
            double b03 = (double)raw[(size_t)band_b03 * band_stride + p] * SCALE;
            double b04 = (double)raw[(size_t)band_b04 * band_stride + p] * SCALE;
            sum += (b02 + b03 + b04) / 3.0;
            total++;
        }
    }

    if (total <= 0) return -1;
    *mean_out = sum / (double)total;
    return 0;
}

/* contrast = stddev((B02+B03+B04)/3) con reflectancia uint16/10000. */
static int block_visible_contrast_std(
    const uint16_t* raw,
    int bands,
    int image_height,
    int image_width,
    int block_size,
    int block_y,
    int block_x,
    int band_b02,
    int band_b03,
    int band_b04,
    double* std_out
) {
    size_t band_stride = (size_t)image_height * (size_t)image_width;
    int y0 = block_y * block_size;
    int x0 = block_x * block_size;
    double sum = 0.0;
    double sum_sq = 0.0;
    int total = 0;
    (void)bands;

    const double SCALE = 1.0 / 10000.0;

    for (int y = y0; y < y0 + block_size; ++y) {
        size_t row_off = (size_t)y * (size_t)image_width;
        for (int x = x0; x < x0 + block_size; ++x) {
            size_t p = row_off + (size_t)x;
            double b02 = (double)raw[(size_t)band_b02 * band_stride + p] * SCALE;
            double b03 = (double)raw[(size_t)band_b03 * band_stride + p] * SCALE;
            double b04 = (double)raw[(size_t)band_b04 * band_stride + p] * SCALE;
            double brightness = (b02 + b03 + b04) / 3.0;
            sum += brightness;
            sum_sq += brightness * brightness;
            total++;
        }
    }

    if (total <= 0) return -1;
    double mean = sum / (double)total;
    double variance = (sum_sq / (double)total) - (mean * mean);
    if (variance < 0.0 && variance > -1e-15) variance = 0.0;
    if (variance < 0.0) return -1;
    *std_out = sqrt(variance);
    return 0;
}

/* ---------- Tipo 2b: Bare Soil Index (BSI) ---------- */
/* BSI = ((B11+B04) - (B08+B02)) / ((B11+B04) + (B08+B02)) */
static int block_bsi_mean(
    const uint16_t* raw,
    int bands,
    int image_height,
    int image_width,
    int block_size,
    int block_y,
    int block_x,
    int band_b02,
    int band_b04,
    int band_b08,
    int band_b11,
    double* mean_out
) {
    size_t band_stride = (size_t)image_height * (size_t)image_width;
    int y0 = block_y * block_size;
    int x0 = block_x * block_size;
    double sum = 0.0;
    int valid = 0;
    (void)bands;

    for (int y = y0; y < y0 + block_size; ++y) {
        size_t row_off = (size_t)y * (size_t)image_width;
        for (int x = x0; x < x0 + block_size; ++x) {
            size_t p = row_off + (size_t)x;
            double v02 = (double)raw[(size_t)band_b02 * band_stride + p];
            double v04 = (double)raw[(size_t)band_b04 * band_stride + p];
            double v08 = (double)raw[(size_t)band_b08 * band_stride + p];
            double v11 = (double)raw[(size_t)band_b11 * band_stride + p];

            double num = (v11 + v04) - (v08 + v02);
            double den = (v11 + v04) + (v08 + v02);
            if (fabs(den) < 1e-12) continue;
            sum += num / den;
            valid++;
        }
    }

    if (valid <= 0) return -1;
    *mean_out = sum / (double)valid;
    return 0;
}

/* ---------- Tipo 2b: Burned Area Index for Sentinel-2 (BAIS2) ---------- */
/* BAIS2 = (1 - sqrt(B06*B07*B8A / B04)) * ((B12-B8A) / sqrt(B12+B8A) + 1) */
static int block_bais2_mean(
    const uint16_t* raw,
    int bands,
    int image_height,
    int image_width,
    int block_size,
    int block_y,
    int block_x,
    int band_b04,
    int band_b06,
    int band_b07,
    int band_b8a,
    int band_b12,
    double* mean_out
) {
    size_t band_stride = (size_t)image_height * (size_t)image_width;
    int y0 = block_y * block_size;
    int x0 = block_x * block_size;
    double sum = 0.0;
    int valid = 0;
    (void)bands;

    /* Sentinel-2 reflectancia: uint16 con factor 10000 */
    const double SCALE = 1.0 / 10000.0;

    for (int y = y0; y < y0 + block_size; ++y) {
        size_t row_off = (size_t)y * (size_t)image_width;
        for (int x = x0; x < x0 + block_size; ++x) {
            size_t p = row_off + (size_t)x;
            double v04 = (double)raw[(size_t)band_b04 * band_stride + p] * SCALE;
            double v06 = (double)raw[(size_t)band_b06 * band_stride + p] * SCALE;
            double v07 = (double)raw[(size_t)band_b07 * band_stride + p] * SCALE;
            double v8a = (double)raw[(size_t)band_b8a * band_stride + p] * SCALE;
            double v12 = (double)raw[(size_t)band_b12 * band_stride + p] * SCALE;

            /* Evitar division por zero o raiz de negativo */
            if (v04 < 1e-12) continue;
            double prod = v06 * v07 * v8a;
            if (prod < 0.0) continue;
            double sum_12_8a = v12 + v8a;
            if (sum_12_8a < 1e-12) continue;

            double term1 = 1.0 - sqrt(prod / v04);
            double term2 = (v12 - v8a) / sqrt(sum_12_8a) + 1.0;
            sum += term1 * term2;
            valid++;
        }
    }

    if (valid <= 0) return -1;
    *mean_out = sum / (double)valid;
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
    printf("Q-map semantico: blocks=%d unique_Q=%d q_min=%d q_max=%d q_mean=%.4f\n",
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

    BandLayout layout = resolve_layout(opt.layout, opt.bands);
    const PresetSpec* spec = spec_for_preset(opt.preset);
    if (!spec) {
        usage(argv[0]);
        return 2;
    }
    double threshold = opt.threshold_set ? opt.threshold : spec->threshold;
    double roi_target_mse = NAN;
    double roi_target_psnr = NAN;
    if (opt.policy == POLICY_TARGET_FOCUS) {
        if (opt.roi_target_psnr_set) {
            roi_target_psnr = opt.roi_target_psnr;
            roi_target_mse = target_mse_from_psnr(opt.roi_target_psnr);
        } else {
            roi_target_mse = opt.roi_target_mse;
            roi_target_psnr = 10.0 * log10((MAX_SAMPLE_VALUE * MAX_SAMPLE_VALUE) / roi_target_mse);
        }
    }
    int needs_raw = (opt.preset != PRESET_UNIFORM && opt.preset != PRESET_MANUAL);

    int q_height = opt.image_height / opt.block_size;
    int q_width = opt.image_width / opt.block_size;
    int count = q_height * q_width;

    FQBlockCalib* blocks = (FQBlockCalib*)calloc((size_t)count, sizeof(FQBlockCalib));
    uint8_t* qmap = (uint8_t*)malloc((size_t)count);
    if (!blocks || !qmap) {
        fprintf(stderr, "Error: memoria insuficiente\n");
        free(blocks);
        free(qmap);
        return 1;
    }

    if (load_calibration(opt.calibration_path, blocks, q_width, q_height) != 0) {
        free(blocks);
        free(qmap);
        return 1;
    }

    BandMap band_map;
    init_band_map(&band_map);
    if (opt.band_map_path) {
        if (load_band_map(opt.band_map_path, opt.bands, &band_map) != 0) {
            free(blocks);
            free(qmap);
            return 1;
        }
        layout = LAYOUT_BAND_MAP;
    }

    uint8_t* roi = NULL;
    if (opt.preset == PRESET_MANUAL) {
        if (opt.roi_map_path) {
            roi = load_roi_map_u8(opt.roi_map_path, q_width, q_height);
        } else if (opt.roi_tsv_path) {
            roi = load_roi_tsv(opt.roi_tsv_path, q_width, q_height);
        } else {
            roi = load_roi_command(opt.roi_command_path, q_width, q_height);
        }
        if (!roi) {
            free(blocks);
            free(qmap);
            return 1;
        }
    }

    uint16_t* raw = NULL;
    if (needs_raw) {
        raw = load_raw_u16(opt.input_path, opt.bands, opt.image_height, opt.image_width);
    }
    if (needs_raw && !raw) {
        free(roi);
        free(blocks);
        free(qmap);
        return 1;
    }

    int band_a = -1;
    int band_b = -1;
    /* Bandas adicionales para evaluaciones multi-banda */
    int band_extra[5] = {-1, -1, -1, -1, -1};
    int bands_ok = 1;
    if (needs_raw) {
        band_a = band_index(spec->a, layout, opt.bands, &band_map);
        band_b = band_index(spec->b, layout, opt.bands, &band_map);
        switch (spec->eval_mode) {
        case EVAL_CLOUD_CBY:
            /* B03=a, B04=b ya resueltos; B11 opcional */
            band_extra[0] = band_index(BAND_B11, layout, opt.bands, &band_map);
            /* B11 < 0 es aceptable: fallback a version basica (73% precision) */
            if (band_extra[0] >= 0) {
                printf("  clouds CBY: B11 disponible -> version mejorada (88%% precision)\n");
            } else {
                printf("  clouds CBY: B11 no disponible -> version basica (73%% precision)\n");
            }
            break;
        case EVAL_VISIBLE_BRIGHTNESS:
        case EVAL_VISIBLE_CONTRAST:
            /* B02=a, B03=b y B04=extra[0] */
            band_extra[0] = band_index(BAND_B04, layout, opt.bands, &band_map);
            if (band_a < 0 || band_b < 0 || band_extra[0] < 0) {
                bands_ok = 0;
            }
            break;
        case EVAL_BSI:
            /* BSI = ((B11+B04) - (B08+B02)) / ((B11+B04) + (B08+B02)) */
            band_extra[0] = band_index(BAND_B02, layout, opt.bands, &band_map);
            band_extra[1] = band_index(BAND_B04, layout, opt.bands, &band_map);
            band_extra[2] = band_index(BAND_B08, layout, opt.bands, &band_map);
            band_extra[3] = band_index(BAND_B11, layout, opt.bands, &band_map);
            if (band_extra[0] < 0 || band_extra[1] < 0 || band_extra[2] < 0 || band_extra[3] < 0) {
                bands_ok = 0;
            }
            break;
        case EVAL_BAIS2:
            /* BAIS2 = (1-sqrt(B06*B07*B8A/B04)) * ((B12-B8A)/sqrt(B12+B8A)+1) */
            band_extra[0] = band_index(BAND_B04, layout, opt.bands, &band_map);
            band_extra[1] = band_index(BAND_B06, layout, opt.bands, &band_map);
            band_extra[2] = band_index(BAND_B07, layout, opt.bands, &band_map);
            band_extra[3] = band_index(BAND_B8A, layout, opt.bands, &band_map);
            band_extra[4] = band_index(BAND_B12, layout, opt.bands, &band_map);
            if (band_extra[0] < 0 || band_extra[1] < 0 || band_extra[2] < 0 ||
                band_extra[3] < 0 || band_extra[4] < 0) {
                bands_ok = 0;
            }
            break;
        case EVAL_NORMALIZED_DIFF:
        default:
            /* Tipo 1: band_a y band_b bastan */
            break;
        }
    }

    double log_mean = 0.0;
    double log_std = 1.0;
    log_mse_stats(blocks, count, &log_mean, &log_std);

    FILE* summary = NULL;
    if (opt.summary_tsv_path) {
        summary = fopen(opt.summary_tsv_path, "w");
        if (!summary) {
            fprintf(stderr, "Error: no se pudo abrir summary TSV '%s'\n", opt.summary_tsv_path);
            free(raw);
            free(roi);
            free(blocks);
            free(qmap);
            return 1;
        }
        fprintf(summary,
                "block_y\tblock_x\tpreset\tindex_name\tindex_mean\tthreshold\ttarget_psnr\ttarget_mse\ttarget_feasibility\tbase_q\ttarget_q\tfinal_q\tsemantic_match\tforeground_boost_applied\tbackground_penalty_applied\tbackground_fixed_q_applied\treason\tq_range_status\n");
    }

    SemanticSummary ss;
    memset(&ss, 0, sizeof(ss));

    for (int by = 0; by < q_height; ++by) {
        for (int bx = 0; bx < q_width; ++bx) {
            int idx = by * q_width + bx;
            int base_q = select_q_adaptive_difficulty(
                &blocks[idx],
                opt.q_mean,
                opt.adaptive_strength,
                log_mean,
                log_std
            );
            int final_q = base_q;
            double index_mean = NAN;
            int match = 0;
            int foreground_applied = 0;
            int background_applied = 0;
            int fixed_q_applied = 0;
            int target_q = -1;
            const char* target_feasibility = "not_requested";
            const char* reason = "uniform_preset";
            const char* q_range_status = "official";

            if (opt.preset == PRESET_MANUAL) {
                ss.semantic_possible++;
                index_mean = (roi[idx] != 0) ? 1.0 : 0.0;
                if (roi[idx] != 0) {
                    ss.semantic_matches++;
                    match = 1;
                    if (opt.policy == POLICY_TARGET_FOCUS) {
                        target_q = select_q_for_target_mse(&blocks[idx], roi_target_mse, &reason, &target_feasibility);
                        final_q = target_q;
                        ss.target_applied++;
                        if (strcmp(reason, "target_model") != 0) ss.target_clamped++;
                    } else if (opt.policy == POLICY_PRESERVE_ROI) {
                        int fixed_q = select_foreground_fixed_q(&opt, &blocks[idx], &q_range_status);
                        final_q = fixed_q;
                        foreground_applied = 1;
                        ss.boost_applied++;
                        reason = (final_q == opt.foreground_q)
                            ? "preserve_roi_foreground_fixed_q"
                            : "preserve_roi_foreground_fixed_q_clamped";
                    } else {
                        int boosted = clamp_int(base_q + opt.foreground_boost, blocks[idx].q_min, blocks[idx].q_max);
                        final_q = boosted;
                        if (final_q > base_q) {
                            foreground_applied = 1;
                            ss.boost_applied++;
                            reason = (opt.policy == POLICY_FOCUS) ? "manual_focus_foreground_boost" : "manual_boost";
                        } else {
                            ss.boost_clamped++;
                            reason = (opt.policy == POLICY_FOCUS) ? "manual_focus_foreground_clamped" : "manual_match_clamped";
                        }
                    }
                } else if (policy_uses_focus_background(opt.policy) && opt.background_q_set) {
                    int fixed_q = select_background_fixed_q(&opt, &blocks[idx], &q_range_status);
                    final_q = fixed_q;
                    if (final_q < base_q) {
                        fixed_q_applied = 1;
                        ss.fixed_q_applied++;
                        reason = (opt.policy == POLICY_PRESERVE_ROI)
                            ? "preserve_roi_background_fixed_q"
                            : "manual_focus_background_fixed_q";
                    } else {
                        ss.fixed_q_clamped++;
                        reason = (opt.policy == POLICY_PRESERVE_ROI)
                            ? "preserve_roi_background_fixed_q_not_lower"
                            : "manual_focus_background_fixed_q_not_lower";
                    }
                } else if (policy_uses_focus_background(opt.policy) && opt.background_penalty > 0) {
                    int penalized = clamp_int(base_q - opt.background_penalty, blocks[idx].q_min, blocks[idx].q_max);
                    final_q = penalized;
                    if (final_q < base_q) {
                        background_applied = 1;
                        ss.penalty_applied++;
                        reason = "manual_focus_background_penalty";
                    } else {
                        ss.penalty_clamped++;
                        reason = "manual_focus_background_clamped";
                    }
                } else {
                    reason = "manual_no_match";
                }
            } else if (opt.preset == PRESET_UNIFORM) {
                reason = "uniform_preset";
            } else if (!bands_ok || (spec->eval_mode == EVAL_NORMALIZED_DIFF && (band_a < 0 || band_b < 0))) {
                ss.missing_bands++;
                reason = "missing_bands";
            } else {
                ss.semantic_possible++;
                int eval_ok = 0;
                switch (spec->eval_mode) {
                case EVAL_CLOUD_CBY:
                    eval_ok = (block_cloud_cby_fraction(
                        raw, opt.bands, opt.image_height, opt.image_width,
                        opt.block_size, by, bx,
                        band_a, band_b, band_extra[0],
                        &index_mean) == 0);
                    break;
                case EVAL_VISIBLE_BRIGHTNESS:
                    eval_ok = (block_visible_brightness_mean(
                        raw, opt.bands, opt.image_height, opt.image_width,
                        opt.block_size, by, bx,
                        band_a, band_b, band_extra[0],
                        &index_mean) == 0);
                    break;
                case EVAL_VISIBLE_CONTRAST:
                    eval_ok = (block_visible_contrast_std(
                        raw, opt.bands, opt.image_height, opt.image_width,
                        opt.block_size, by, bx,
                        band_a, band_b, band_extra[0],
                        &index_mean) == 0);
                    break;
                case EVAL_BSI:
                    eval_ok = (block_bsi_mean(
                        raw, opt.bands, opt.image_height, opt.image_width,
                        opt.block_size, by, bx,
                        band_extra[0], band_extra[1], band_extra[2], band_extra[3],
                        &index_mean) == 0);
                    break;
                case EVAL_BAIS2:
                    eval_ok = (block_bais2_mean(
                        raw, opt.bands, opt.image_height, opt.image_width,
                        opt.block_size, by, bx,
                        band_extra[0], band_extra[1], band_extra[2], band_extra[3], band_extra[4],
                        &index_mean) == 0);
                    break;
                case EVAL_NORMALIZED_DIFF:
                default:
                    eval_ok = (block_index_mean(
                        raw, opt.bands, opt.image_height, opt.image_width,
                        opt.block_size, by, bx,
                        band_a, band_b,
                        &index_mean) == 0);
                    break;
                }
                if (!eval_ok) {
                    ss.no_valid_pixels++;
                    reason = "no_valid_pixels";
                } else if (semantic_match(index_mean, threshold, spec->cmp)) {
                    ss.semantic_matches++;
                    match = 1;
                    if (opt.policy == POLICY_TARGET_FOCUS) {
                        target_q = select_q_for_target_mse(&blocks[idx], roi_target_mse, &reason, &target_feasibility);
                        final_q = target_q;
                        ss.target_applied++;
                        if (strcmp(reason, "target_model") != 0) ss.target_clamped++;
                    } else if (opt.policy == POLICY_PRESERVE_ROI) {
                        int fixed_q = select_foreground_fixed_q(&opt, &blocks[idx], &q_range_status);
                        final_q = fixed_q;
                        foreground_applied = 1;
                        ss.boost_applied++;
                        reason = (final_q == opt.foreground_q)
                            ? "preserve_roi_foreground_fixed_q"
                            : "preserve_roi_foreground_fixed_q_clamped";
                    } else {
                        int boosted = clamp_int(base_q + opt.foreground_boost, blocks[idx].q_min, blocks[idx].q_max);
                        final_q = boosted;
                        if (final_q > base_q) {
                            foreground_applied = 1;
                            ss.boost_applied++;
                            reason = (opt.policy == POLICY_FOCUS) ? "focus_foreground_boost" : "semantic_boost";
                        } else {
                            ss.boost_clamped++;
                            reason = (opt.policy == POLICY_FOCUS) ? "focus_foreground_clamped" : "semantic_match_clamped";
                        }
                    }
                } else {
                    if (policy_uses_focus_background(opt.policy) && opt.background_q_set) {
                        int fixed_q = select_background_fixed_q(&opt, &blocks[idx], &q_range_status);
                        final_q = fixed_q;
                        if (final_q < base_q) {
                            fixed_q_applied = 1;
                            ss.fixed_q_applied++;
                            reason = (opt.policy == POLICY_PRESERVE_ROI)
                                ? "preserve_roi_background_fixed_q"
                                : "focus_background_fixed_q";
                        } else {
                            ss.fixed_q_clamped++;
                            reason = (opt.policy == POLICY_PRESERVE_ROI)
                                ? "preserve_roi_background_fixed_q_not_lower"
                                : "focus_background_fixed_q_not_lower";
                        }
                    } else if (policy_uses_focus_background(opt.policy) && opt.background_penalty > 0) {
                        int penalized = clamp_int(base_q - opt.background_penalty, blocks[idx].q_min, blocks[idx].q_max);
                        final_q = penalized;
                        if (final_q < base_q) {
                            background_applied = 1;
                            ss.penalty_applied++;
                            reason = "focus_background_penalty";
                        } else {
                            ss.penalty_clamped++;
                            reason = "focus_background_clamped";
                        }
                    } else {
                        reason = "semantic_no_match";
                    }
                }
            }

            qmap[idx] = (uint8_t)final_q;
            if (summary) {
                if (isnan(index_mean)) {
                    fprintf(summary, "%d\t%d\t%s\t%s\tnan\t%.9g\t%.9g\t%.9g\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%s\t%s\n",
                            by, bx, spec->name, spec->index_name, threshold,
                            roi_target_psnr, roi_target_mse, target_feasibility,
                            base_q, target_q, final_q,
                            match, foreground_applied, background_applied, fixed_q_applied, reason, q_range_status);
                } else {
                    fprintf(summary, "%d\t%d\t%s\t%s\t%.9g\t%.9g\t%.9g\t%.9g\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%s\t%s\n",
                            by, bx, spec->name, spec->index_name, index_mean, threshold,
                            roi_target_psnr, roi_target_mse, target_feasibility,
                            base_q, target_q, final_q,
                            match, foreground_applied, background_applied, fixed_q_applied, reason, q_range_status);
                }
            }
        }
    }

    if (summary) fclose(summary);

    if (write_qmap(opt.output_qmap_path, qmap, (size_t)count) != 0) {
        free(raw);
        free(roi);
        free(blocks);
        free(qmap);
        return 1;
    }

    printf("Preset: %s index=%s threshold=%.6f policy=%s foreground_boost=%d background_penalty=%d",
           spec->name,
           spec->index_name,
           threshold,
           policy_name(opt.policy),
           opt.foreground_boost,
           opt.background_penalty);
    if (opt.policy == POLICY_PRESERVE_ROI || opt.foreground_q_set) {
        printf(" foreground_q=%d", opt.foreground_q);
    }
    if (opt.background_q_set) {
        printf(" background_q=%d", opt.background_q);
        if (opt.allow_experimental_low_q) {
            printf(" experimental_low_q=1");
        }
    }
    if (opt.policy == POLICY_TARGET_FOCUS) {
        printf(" roi_target_psnr=%.6f roi_target_mse=%.9g", roi_target_psnr, roi_target_mse);
    }
    printf("\n");
    printf("Entrada: bands=%d height=%d width=%d block=%d layout=%s\n",
           opt.bands, opt.image_height, opt.image_width, opt.block_size, layout_name(layout));
    if (opt.band_map_path) {
        printf("Band-map: %s\n", opt.band_map_path);
    }
    if (opt.preset == PRESET_MANUAL) {
        const char* roi_source = opt.roi_map_path ? opt.roi_map_path : (opt.roi_tsv_path ? opt.roi_tsv_path : opt.roi_command_path);
        printf("ROI manual: %s\n", roi_source);
    } else if (needs_raw) {
        switch (spec->eval_mode) {
        case EVAL_CLOUD_CBY:
            printf("Bandas CBY: B03=%d B04=%d B11=%d%s\n",
                   band_a, band_b, band_extra[0],
                   band_extra[0] < 0 ? " (no disponible, version basica)" : " (version mejorada)");
            break;
        case EVAL_VISIBLE_BRIGHTNESS:
            printf("Bandas VIS_MEAN: B02=%d B03=%d B04=%d%s\n",
                   band_a, band_b, band_extra[0],
                   bands_ok ? "" : " (bandas insuficientes)");
            break;
        case EVAL_VISIBLE_CONTRAST:
            printf("Bandas VIS_STD: B02=%d B03=%d B04=%d%s\n",
                   band_a, band_b, band_extra[0],
                   bands_ok ? "" : " (bandas insuficientes)");
            break;
        case EVAL_BSI:
            printf("Bandas BSI: B02=%d B04=%d B08=%d B11=%d%s\n",
                   band_extra[0], band_extra[1], band_extra[2], band_extra[3],
                   bands_ok ? "" : " (bandas insuficientes)");
            break;
        case EVAL_BAIS2:
            printf("Bandas BAIS2: B04=%d B06=%d B07=%d B8A=%d B12=%d%s\n",
                   band_extra[0], band_extra[1], band_extra[2], band_extra[3], band_extra[4],
                   bands_ok ? "" : " (bandas insuficientes)");
            break;
        case EVAL_NORMALIZED_DIFF:
        default:
            printf("Bandas indice: %s=%d %s=%d\n",
                   BAND_NAMES[(int)spec->a], band_a, BAND_NAMES[(int)spec->b], band_b);
            break;
        }
    }
    printf("Base adaptativa: q_mean=%d strength=%.6f log_mse_mean=%.6f log_mse_std=%.6f\n",
           opt.q_mean, opt.adaptive_strength, log_mean, log_std);
    print_q_summary(qmap, count);
    printf("Semantica: possible=%d matches=%d boost_applied=%d boost_clamped=%d penalty_applied=%d penalty_clamped=%d fixed_q_applied=%d fixed_q_clamped=%d target_applied=%d target_clamped=%d missing_bands=%d no_valid_pixels=%d\n",
           ss.semantic_possible,
           ss.semantic_matches,
           ss.boost_applied,
           ss.boost_clamped,
           ss.penalty_applied,
           ss.penalty_clamped,
           ss.fixed_q_applied,
           ss.fixed_q_clamped,
           ss.target_applied,
           ss.target_clamped,
           ss.missing_bands,
           ss.no_valid_pixels);
    printf("Salida: %s\n", opt.output_qmap_path);

    free(raw);
    free(roi);
    free(blocks);
    free(qmap);
    return 0;
}
