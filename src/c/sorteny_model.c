#include "sorteny_model.h"
#include <string.h> // Para strcmp, strtok, sprintf


// --- Constantes Internas ---
#define LINE_BUFFER_SIZE 1024 // Tamaño máximo para una línea del TSV
#define PATH_BUFFER_SIZE 256  // Tamaño máximo para una ruta de archivo

/**
 * @brief Función auxiliar interna para cargar un único tensor desde un .bin.
 * Abre el archivo, reserva la memoria exacta y lee los datos.
 */
static float* load_tensor_from_file(const char* base_path, const char* filename, size_t expected_bytes) {
    char file_path[PATH_BUFFER_SIZE];
    snprintf(file_path, sizeof(file_path), "%s/%s", base_path, filename);

    FILE* f = fopen(file_path, "rb");
    if (!f) {
        fprintf(stderr, "Error: No se pudo abrir el archivo de peso: %s\n", file_path);
        return NULL;
    }

    // Reservar memoria
    float* data_ptr = (float*)malloc(expected_bytes);
    if (!data_ptr) {
        fprintf(stderr, "Error: Fallo de Malloc al reservar %zu bytes para %s\n", expected_bytes, filename);
        fclose(f);
        return NULL;
    }

    // Leer los datos binarios
    size_t bytes_read = fread(data_ptr, 1, expected_bytes, f);
    if (bytes_read != expected_bytes) {
        fprintf(stderr, "Error: Lectura incompleta de %s. Esperados %zu, leídos %zu\n",
                filename, expected_bytes, bytes_read);
        fclose(f);
        free(data_ptr);
        return NULL;
    }

    fclose(f);
    return data_ptr;
}



// --- Utilidades de parseo ---

/**
 * @brief Comprueba el tamaño en bytes de un tipo de dato 
 * de momento solo soporta "float32"
 */
static int dtype_sizeof(const char* dtype, size_t* out) {
    if (!dtype || !out) return -1;
    // Solo soportamos float32 por ahora (lo que exporta el script)
    if (strcmp(dtype, "float32") == 0) { *out = sizeof(float); return 0; }
    fprintf(stderr, "Error: dtype no soportado: %s (se esperaba float32)\n", dtype);
    return -1;
}



/**
 * @brief Analiza una cadena de forma y la convierte en un array de dimensiones.
 * 
 * Convierte una cadena con formato de forma (p.ej., "5x5x1x128") en un array de dimensiones size_t.
 * El separador puede ser 'x' o 'X'.
 * 
 * @param shape_str Cadena de entrada que contiene dimensiones separadas por 'x' (p.ej., "5x5x1x128")
 * @param dims Array de salida para almacenar las dimensiones analizadas
 * @param max_dims Número máximo de dimensiones que se pueden almacenar en el array dims
 * @param out_ndims Puntero para almacenar el número real de dimensiones analizadas
 * 
 * @return 0 en caso de éxito, -1 en caso de error (entrada inválida, no se analizaron dimensiones, o fallo en el análisis)
 * 
 * @example
 * size_t dims[4];
 * size_t ndims;
 * parse_shape_dims("5x5x1x128", dims, 4, &ndims);
 * // Resultado: dims = {5, 5, 1, 128}, ndims = 4
 */
static int parse_shape_dims(const char* shape_str, size_t* dims, size_t max_dims, size_t* out_ndims) {
    if (!shape_str || !dims || !out_ndims) return -1;
    char buf[LINE_BUFFER_SIZE];
    strncpy(buf, shape_str, sizeof(buf));
    buf[sizeof(buf)-1] = '\0';
    // El TSV usa separador 'x' (p.ej. 5x5x1x128)
    size_t count = 0;
    char* p = buf;
    while (*p && count < max_dims) {
        // parse número
        char* end = NULL;
        dims[count] = (size_t)strtoull(p, &end, 10);
        if (end == p) return -1; // no avanzo
        count++;
        if (*end == 'x' || *end == 'X') {
            p = end + 1; // siguiente token
        } else {
            // Si termina en newline u otros, salimos
            break;
        }
    }
    *out_ndims = count;
    return (count > 0) ? 0 : -1;
}



/**
 * @brief Calcula el tamaño esperado en bytes de un tensor dado su dtype y shape.
 * 
 * @param dtype Tipo de dato (p.ej., "float32")
 * @param shape_str Cadena de forma (p.ej., "5x5x1x128")
 * @param out_bytes Puntero para almacenar el tamaño esperado en bytes
 * 
 * @return 0 en caso de éxito, -1 en caso de error
 */
static int compute_expected_bytes(const char* dtype, const char* shape_str, size_t* out_bytes) {
    size_t sz; if (dtype_sizeof(dtype, &sz) != 0) return -1;
    size_t dims[8]; size_t nd=0;
    if (parse_shape_dims(shape_str, dims, 8, &nd) != 0) return -1;
    // Producto de dimensiones (unsigned __int128 es extensión GCC para evitar overflow)
    unsigned __int128 prod = 1;
    for (size_t i = 0; i < nd; ++i) prod *= dims[i];
    // Convertir a size_t si cabe
    size_t elems = (size_t)prod;
    *out_bytes = elems * sz;
    return 0;
}
/**
 * @brief Transpone el layout de memoria de [kH][kW][DimA][DimB] a [kH][kW][DimB][DimA].
 * Necesario porque los pesos ieec050 tienen layout [Out][In] (o viceversa) respecto a lo esperado [In][Out].
 */
static void transpose_kernel_layout(float* kernel, size_t kH, size_t kW, size_t dimA, size_t dimB) {
    size_t matrix_size = dimA * dimB;
    size_t k_count = kH * kW;
    float* temp = (float*)malloc(matrix_size * sizeof(float));
    if (!temp) {
        fprintf(stderr, "Error: Fallo de malloc en transpose_kernel_layout\n");
        return;
    }

    for (size_t k = 0; k < k_count; k++) {
        float* matrix_ptr = kernel + (k * matrix_size);
        // Copiar a temp
        memcpy(temp, matrix_ptr, matrix_size * sizeof(float));
        // Transponer de temp(DimA, DimB) a matrix_ptr(DimB, DimA)
        // temp[i*DimB + j] -> matrix_ptr[j*DimA + i]
        for (size_t i = 0; i < dimA; i++) {
            for (size_t j = 0; j < dimB; j++) {
                matrix_ptr[j * dimA + i] = temp[i * dimB + j];
            }
        }
    }
    free(temp);
}


/**
 * @brief Lee el 'weights_index.tsv' y puebla la estructura SORTENY_Model.
 */
SORTENY_Model* load_model_weights(const char* base_path) {
    int conv_dims_swapped = (base_path && strstr(base_path, "transposed") != NULL);
    char index_path[PATH_BUFFER_SIZE];
    snprintf(index_path, sizeof(index_path), "%s/weights_index.tsv", base_path);

    FILE* f_index = fopen(index_path, "r");
    if (!f_index) {
        fprintf(stderr, "Error: No se pudo abrir 'weights_index.tsv' en %s\n", base_path);
        return NULL;
    }

    // 1. Reservar memoria para la estructura principal del modelo
    SORTENY_Model* model = (SORTENY_Model*)malloc(sizeof(SORTENY_Model));
    if (!model) {
        fprintf(stderr, "Error: Fallo de Malloc para SORTENY_Model\n");
        fclose(f_index);
        return NULL;
    }
    // Inicializar todos los punteros a NULL para una liberación segura en caso de error
    memset(model, 0, sizeof(SORTENY_Model));

    char line[LINE_BUFFER_SIZE];
    
    // 2. Omitir la línea de cabecera del TSV
    if (fgets(line, sizeof(line), f_index) == NULL) {
        fprintf(stderr, "Error: Archivo 'weights_index.tsv' está vacío.\n");
        fclose(f_index);
        free_model_weights(model); // Libera el 'model' antes de salir
        return NULL;
    }

    // 3. Leer y procesar cada línea de pesos
    while (fgets(line, sizeof(line), f_index) != NULL) {
        // Formato TSV: filename \t dtype \t size_bytes \t shape \t sha256
        char* filename = strtok(line, "\t");
        char* dtype = strtok(NULL, "\t");
        char* size_bytes_str = strtok(NULL, "\t");
        char* shape_str = strtok(NULL, "\t");
        // (sha256 no lo usamos aquí)

        if (!filename || !dtype || !size_bytes_str || !shape_str) continue; // Línea mal formada

        // Calcular expected_bytes desde dtype y shape (y compararlo con size_bytes)
        size_t declared_bytes = (size_t)strtoull(size_bytes_str, NULL, 10);
        size_t expected_bytes = 0;
        if (compute_expected_bytes(dtype, shape_str, &expected_bytes) != 0) {
            fprintf(stderr, "Error: No se pudo calcular expected_bytes para %s (dtype=%s, shape=%s)\n", filename, dtype, shape_str);
            free_model_weights(model);
            fclose(f_index);
            return NULL;
        }
        if (declared_bytes != expected_bytes) {
            fprintf(stderr, "Aviso: size_bytes declarado (%zu) difiere del esperado (%zu) para %s. Usando el esperado.\n",
                    declared_bytes, expected_bytes, filename);
        }
        float* data_ptr = load_tensor_from_file(base_path, filename, expected_bytes);
        
        if (data_ptr == NULL) {
            free_model_weights(model); // Liberar toda la memoria si un archivo falla
            fclose(f_index);
            return NULL;
        }

        // --- MAPEO DE NOMBRES A ESTRUCTURAS ---
        // Este bloque es el "corazón" del cargador.
        // Utiliza el 'weights_index.tsv' como referencia.
        
        // 1. Spectral Analysis
        if (strcmp(filename, "spectral_analysis_kernel.bin") == 0 ||
            strcmp(filename, "spectral__layer_with_weights-0_kernel_.ATTRIBUTES_VARIABLE_VALUE.bin") == 0) {
            model->spectral_an.dense.kernel = data_ptr;
            size_t dims[4], nd=0; if (parse_shape_dims(shape_str, dims, 4, &nd) == 0 && nd == 2) {
                // TSV Shape is [Cin, Cout]
                model->spectral_an.dense.C_in = dims[0];
                model->spectral_an.dense.C_out = dims[1];
            }
        }
        
        // 2. Analysis Transform
        else if (strcmp(filename, "analysis_conv_0_kernel.bin") == 0 ||
                 strcmp(filename, "analysis__variables_3_.ATTRIBUTES_VARIABLE_VALUE.bin") == 0 ||
                 strcmp(filename, "analysis__variables_4_.ATTRIBUTES_VARIABLE_VALUE.bin") == 0) {
            model->analysis_an.conv_0.kernel = data_ptr;
            size_t dims[4], nd=0; if (parse_shape_dims(shape_str, dims, 4, &nd) == 0 && nd == 4) {
                // TSV Shape is [kH, kW, Cin, Cout]
                model->analysis_an.conv_0.kH = dims[0];
                model->analysis_an.conv_0.kW = dims[1];
                if (conv_dims_swapped) {
                    model->analysis_an.conv_0.C_in = dims[3];
                    model->analysis_an.conv_0.C_out = dims[2];
                    transpose_kernel_layout(data_ptr, dims[0], dims[1], dims[2], dims[3]);
                } else {
                    model->analysis_an.conv_0.C_in = dims[2]; 
                    model->analysis_an.conv_0.C_out = dims[3];
                }
            }
            model->analysis_an.conv_0.stride = 2;
        } else if (strcmp(filename, "analysis_conv_0_bias.bin") == 0 ||
                   strcmp(filename, "analysis__layer_with_weights-0__bias_parameter_.ATTRIBUTES_VARIABLE_VALUE.bin") == 0) {
            model->analysis_an.conv_0.bias = data_ptr;
            model->analysis_an.conv_0.has_bias = 1;
        } else if (strcmp(filename, "analysis_gdn_0_beta.bin") == 0 ||
                   strcmp(filename, "analysis__variables_1_.ATTRIBUTES_VARIABLE_VALUE.bin") == 0) {
            model->analysis_an.gdn_0.beta = data_ptr;
            size_t dims[4], nd=0;
            if (parse_shape_dims(shape_str, dims, 4, &nd) == 0 && nd >= 1) {
                model->analysis_an.gdn_0.C = dims[0];
                /* Usamos los valores exportados directamente (ya son efectivos). */
            }
            model->analysis_an.gdn_0.epsilon = 1.0f;
        } else if (strcmp(filename, "analysis_gdn_0_gamma.bin") == 0 ||
                   strcmp(filename, "analysis__variables_2_.ATTRIBUTES_VARIABLE_VALUE.bin") == 0) {
            model->analysis_an.gdn_0.gamma = data_ptr;
            /* Valores gamma ya efectivos, no elevar al cuadrado. */
            model->analysis_an.gdn_0.epsilon = 1.0f;
        } 
        
        else if (strcmp(filename, "analysis_conv_1_kernel.bin") == 0 ||
                 strcmp(filename, "analysis__variables_8_.ATTRIBUTES_VARIABLE_VALUE.bin") == 0 ||
                 strcmp(filename, "analysis__variables_9_.ATTRIBUTES_VARIABLE_VALUE.bin") == 0) {
            model->analysis_an.conv_1.kernel = data_ptr;
            size_t dims[4], nd=0; if (parse_shape_dims(shape_str, dims, 4, &nd) == 0 && nd == 4) {
                model->analysis_an.conv_1.kH = dims[0];
                model->analysis_an.conv_1.kW = dims[1];
                if (conv_dims_swapped) {
                    model->analysis_an.conv_1.C_in = dims[3];
                    model->analysis_an.conv_1.C_out = dims[2];
                    transpose_kernel_layout(data_ptr, dims[0], dims[1], dims[2], dims[3]);
                } else {
                    model->analysis_an.conv_1.C_in = dims[2];
                    model->analysis_an.conv_1.C_out = dims[3];
                }
            }
            model->analysis_an.conv_1.stride = 2;
        } else if (strcmp(filename, "analysis_conv_1_bias.bin") == 0 ||
                   strcmp(filename, "analysis__layer_with_weights-1__bias_parameter_.ATTRIBUTES_VARIABLE_VALUE.bin") == 0) {
            model->analysis_an.conv_1.bias = data_ptr;
            model->analysis_an.conv_1.has_bias = 1;
        } else if (strcmp(filename, "analysis_gdn_1_beta.bin") == 0 ||
                   strcmp(filename, "analysis__variables_6_.ATTRIBUTES_VARIABLE_VALUE.bin") == 0) {
            model->analysis_an.gdn_1.beta = data_ptr;
            size_t dims[4], nd=0;
            if (parse_shape_dims(shape_str, dims, 4, &nd) == 0 && nd >= 1) {
                model->analysis_an.gdn_1.C = dims[0];
            }
            model->analysis_an.gdn_1.epsilon = 1.0f;
        } else if (strcmp(filename, "analysis_gdn_1_gamma.bin") == 0 ||
                   strcmp(filename, "analysis__variables_7_.ATTRIBUTES_VARIABLE_VALUE.bin") == 0) {
            model->analysis_an.gdn_1.gamma = data_ptr;
            model->analysis_an.gdn_1.epsilon = 1.0f;
        }

        else if (strcmp(filename, "analysis_conv_2_kernel.bin") == 0 ||
                 strcmp(filename, "analysis__variables_13_.ATTRIBUTES_VARIABLE_VALUE.bin") == 0 ||
                 strcmp(filename, "analysis__variables_14_.ATTRIBUTES_VARIABLE_VALUE.bin") == 0) {
            model->analysis_an.conv_2.kernel = data_ptr;
            size_t dims[4], nd=0; if (parse_shape_dims(shape_str, dims, 4, &nd) == 0 && nd == 4) {
                model->analysis_an.conv_2.kH = dims[0];
                model->analysis_an.conv_2.kW = dims[1];
                if (conv_dims_swapped) {
                    model->analysis_an.conv_2.C_in = dims[3];
                    model->analysis_an.conv_2.C_out = dims[2];
                    transpose_kernel_layout(data_ptr, dims[0], dims[1], dims[2], dims[3]);
                } else {
                    model->analysis_an.conv_2.C_in = dims[2];
                    model->analysis_an.conv_2.C_out = dims[3];
                }
            }
            model->analysis_an.conv_2.stride = 2;
        } else if (strcmp(filename, "analysis_conv_2_bias.bin") == 0 ||
                   strcmp(filename, "analysis__layer_with_weights-2__bias_parameter_.ATTRIBUTES_VARIABLE_VALUE.bin") == 0) {
            model->analysis_an.conv_2.bias = data_ptr;
            model->analysis_an.conv_2.has_bias = 1;
        } else if (strcmp(filename, "analysis_gdn_2_beta.bin") == 0 ||
                   strcmp(filename, "analysis__variables_11_.ATTRIBUTES_VARIABLE_VALUE.bin") == 0) {
            model->analysis_an.gdn_2.beta = data_ptr;
            size_t dims[4], nd=0;
            if (parse_shape_dims(shape_str, dims, 4, &nd) == 0 && nd >= 1) {
                model->analysis_an.gdn_2.C = dims[0];
            }
            model->analysis_an.gdn_2.epsilon = 1.0f;
        } else if (strcmp(filename, "analysis_gdn_2_gamma.bin") == 0 ||
                   strcmp(filename, "analysis__variables_12_.ATTRIBUTES_VARIABLE_VALUE.bin") == 0) {
            model->analysis_an.gdn_2.gamma = data_ptr;
            model->analysis_an.gdn_2.epsilon = 1.0f;
        }

        else if (strcmp(filename, "analysis_conv_3_kernel.bin") == 0 ||
                 strcmp(filename, "analysis__variables_15_.ATTRIBUTES_VARIABLE_VALUE.bin") == 0 ||
                 strcmp(filename, "analysis__variables_16_.ATTRIBUTES_VARIABLE_VALUE.bin") == 0) {
            model->analysis_an.conv_3.kernel = data_ptr;
            model->analysis_an.conv_3.has_bias = 0; // No tiene bias
            model->analysis_an.conv_3.bias = NULL;
            size_t dims[4], nd=0; if (parse_shape_dims(shape_str, dims, 4, &nd) == 0 && nd == 4) {
                model->analysis_an.conv_3.kH = dims[0];
                model->analysis_an.conv_3.kW = dims[1];
                if (conv_dims_swapped) {
                    model->analysis_an.conv_3.C_in = dims[3];
                    model->analysis_an.conv_3.C_out = dims[2];
                    transpose_kernel_layout(data_ptr, dims[0], dims[1], dims[2], dims[3]);
                } else {
                    model->analysis_an.conv_3.C_in = dims[2];
                    model->analysis_an.conv_3.C_out = dims[3];
                }
            }
            model->analysis_an.conv_3.stride = 2;
        }

        // 3. Modulating Transform
        else if (strcmp(filename, "mod_dense_1_kernel.bin") == 0 ||
                 strcmp(filename, "modulating__layer_with_weights-0_kernel_.ATTRIBUTES_VARIABLE_VALUE.bin") == 0) {
            model->modulating_mod.dense_0.kernel = data_ptr;
            size_t dims[4], nd=0; if (parse_shape_dims(shape_str, dims, 4, &nd) == 0 && nd == 2) {
                // TSV Shape is [Cin, Cout]
                model->modulating_mod.dense_0.C_in = dims[0];
                model->modulating_mod.dense_0.C_out = dims[1];
            }
        } else if (strcmp(filename, "mod_dense_1_bias.bin") == 0 ||
                   strcmp(filename, "modulating__layer_with_weights-0_bias_.ATTRIBUTES_VARIABLE_VALUE.bin") == 0) { 
             model->modulating_mod.dense_0.bias = data_ptr;
        }
        else if (strcmp(filename, "mod_dense_2_kernel.bin") == 0 ||
                 strcmp(filename, "modulating__layer_with_weights-1_kernel_.ATTRIBUTES_VARIABLE_VALUE.bin") == 0) {
            model->modulating_mod.dense_1.kernel = data_ptr;
            size_t dims[4], nd=0; if (parse_shape_dims(shape_str, dims, 4, &nd) == 0 && nd == 2) {
                // TSV Shape is [Cin, Cout]
                model->modulating_mod.dense_1.C_in = dims[0];
                model->modulating_mod.dense_1.C_out = dims[1];
            }
        } else if (strcmp(filename, "mod_dense_2_bias.bin") == 0 ||
                   strcmp(filename, "modulating__layer_with_weights-1_bias_.ATTRIBUTES_VARIABLE_VALUE.bin") == 0) {
            model->modulating_mod.dense_1.bias = data_ptr;
        }

        else {
            // Si el TSV tiene un peso que no conocemos, lo liberamos para evitar fugas
            fprintf(stderr, "Aviso: Peso '%s' no mapeado. Liberando memoria.\n", filename);
            free(data_ptr);
        }
    }

    fclose(f_index);
    printf("Modelo cargado exitosamente en memoria.\n");
    return model;
}

/**
 * @brief Libera toda la memoria (free) reservada por load_model_weights.
 */
void free_model_weights(SORTENY_Model* model) {
    if (!model) return;

    // 1. Spectral
    free(model->spectral_an.dense.kernel);

    // 2. Analysis
    free(model->analysis_an.conv_0.kernel);
    free(model->analysis_an.conv_0.bias);
    free(model->analysis_an.gdn_0.beta);
    free(model->analysis_an.gdn_0.gamma);
    free(model->analysis_an.conv_1.kernel);
    free(model->analysis_an.conv_1.bias);
    free(model->analysis_an.gdn_1.beta);
    free(model->analysis_an.gdn_1.gamma);
    free(model->analysis_an.conv_2.kernel);
    free(model->analysis_an.conv_2.bias);
    free(model->analysis_an.gdn_2.beta);
    free(model->analysis_an.gdn_2.gamma);
    free(model->analysis_an.conv_3.kernel);
    
    // 3. Modulating
    free(model->modulating_mod.dense_0.kernel);
    free(model->modulating_mod.dense_0.bias);
    free(model->modulating_mod.dense_1.kernel);
    free(model->modulating_mod.dense_1.bias);

    // Finalmente, liberar la estructura principal
    free(model);
}