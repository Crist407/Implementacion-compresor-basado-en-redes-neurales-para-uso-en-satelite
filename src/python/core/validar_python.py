import tensorflow as tf
import tensorflow_compression as tfc
import numpy as np
import os

# --- Configuración ---
MODEL_DIR = "models/SORTENY_Sentinel2_model"
IMAGE_FILE = "data/T31TCG_20230907T104629_5.8_512_512_2_1_0.raw"
LAMBDA_VAL = 0.01
OUTPUT_FILE = "python_ground_truth.bin" 

# --- Dimensiones ---
BANDS = 8
H_IN = 512
W_IN = 512
C3_AN = 384
# Capas intermedias Analysis (para los volcados)
C0 = 128
H1 = H_IN // 2   # 256
W1 = W_IN // 2   # 256
H2 = H1 // 2     # 128
W2 = W1 // 2     # 128
H3 = H2 // 2     # 64
W3 = W2 // 2     # 64
H4 = H_IN // 16  # 32
W4 = W_IN // 16  # 32

# Alias global para la carga del modelo Keras
bit_length = 16

# --- INICIO: Clases personalizadas ---

class ModulatingTransform(tf.keras.Sequential):
  def __init__(self, hidden_nodes, num_filters, maxval):
    super().__init__()
    self.add(tf.keras.layers.Lambda(lambda x: x / maxval))
    self.add(tf.keras.layers.Dense(hidden_nodes, activation=tf.nn.relu, kernel_initializer='ones'))
    self.add(tf.keras.layers.Dense(num_filters, activation=tf.nn.relu, kernel_initializer='ones'))

class AnalysisTransform(tf.keras.Sequential):
  def __init__(self, num_filters_hidden, num_filters_latent):
    super().__init__(name="analysis")
    self.add(tf.keras.layers.Lambda(lambda x: x / ((2**bit_length)-1)))
    self.add(tfc.SignalConv2D(
        num_filters_hidden, (5, 5), name="layer_0", corr=True, strides_down=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="gdn_0")))
    self.add(tfc.SignalConv2D(
        num_filters_hidden, (5, 5), name="layer_1", corr=True, strides_down=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="gdn_1")))
    self.add(tfc.SignalConv2D(
        num_filters_hidden, (5, 5), name="layer_2", corr=True, strides_down=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="gdn_2"))) 
    self.add(tfc.SignalConv2D(
        num_filters_latent, (5, 5), name="layer_3", corr=True, strides_down=2,
        padding="same_zeros", use_bias=False,
        activation=None))

class SynthesisTransform(tf.keras.Sequential):
  def __init__(self, num_filters):
    super().__init__(name="synthesis")
    self.add(tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2)))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_0", corr=False, strides_up=1,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="igdn_0", inverse=True)))
    self.add(tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2)))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_1", corr=False, strides_up=1,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="igdn_1", inverse=True)))
    self.add(tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2)))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_2", corr=False, strides_up=1,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="igdn_2", inverse=True)))
    self.add(tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2)))
    self.add(tfc.SignalConv2D(
        1, (5, 5), name="layer_3", corr=False, strides_up=1,
        padding="same_zeros", use_bias=True,
        activation=None))
    self.add(tf.keras.layers.Lambda(lambda x: x * ((2**bit_length)-1)))

class HyperAnalysisTransform(tf.keras.Sequential):
  def __init__(self, num_filters_hidden_hyperprior, num_filters_latent_hyperprior):
    super().__init__(name="hyper_analysis")
    self.add(tfc.SignalConv2D(
        num_filters_hidden_hyperprior, (3, 3), name="layer_0", corr=True, strides_down=1,
        padding="same_zeros", use_bias=True,
        activation=tf.nn.relu))
    self.add(tfc.SignalConv2D(
        num_filters_hidden_hyperprior, (5, 5), name="layer_1", corr=True, strides_down=2,
        padding="same_zeros", use_bias=True,
        activation=tf.nn.relu))
    self.add(tfc.SignalConv2D(
        num_filters_latent_hyperprior, (5, 5), name="layer_2", corr=True, strides_down=2,
        padding="same_zeros", use_bias=False,
        activation=None))

class HyperSynthesisTransform(tf.keras.Sequential):
  def __init__(self, num_filters_hidden_hyperprior, num_filters_latent):
    super().__init__(name="hyper_synthesis")
    # (Definición completa no es necesaria para la compresión)
    pass
    
class SpectralAnalysisTransform(tf.keras.Sequential):
  def __init__(self, num_filters_1D, init):
    super().__init__(name="spectral_analysis")
    self.add(tf.keras.layers.Dense(num_filters_1D, activation=None, use_bias=False, kernel_initializer=init))
# --- FIN: Clases personalizadas ---


def load_image_raw_u16_bsq_to_nhwc(filename, bands, height, width):
    """Carga la imagen raw (uint16, BSQ) y la convierte a float32 (1, H, W, B)."""
    total_elements = bands * height * width
    bsq_buffer = np.fromfile(filename, dtype=np.uint16, count=total_elements)
    if bsq_buffer.size != total_elements:
        raise IOError(f"Error: Lectura incompleta de la imagen: {filename}")
    
    img_bsq = bsq_buffer.reshape((bands, height, width))
    
    # Transponer a NHWC (1, H, W, B) y convertir a float32
    img_nhwc = np.transpose(img_bsq, (1, 2, 0)).astype(np.float32)
    img_batch = np.expand_dims(img_nhwc, axis=0) # Añadir batch dim
    return img_batch

def main():
    print(f"Cargando modelo '{MODEL_DIR}'...")
    model = tf.keras.models.load_model(MODEL_DIR, compile=False)
    print("Modelo cargado.")

    print(f"Cargando imagen '{IMAGE_FILE}'...")
    image_tensor = load_image_raw_u16_bsq_to_nhwc(IMAGE_FILE, BANDS, H_IN, W_IN)
    print(f"Imagen cargada, forma: {image_tensor.shape}") # (1, 512, 512, 8)

    lambda_tensor = tf.constant([[[[LAMBDA_VAL]]]], dtype=tf.float32)

    # --- Replicar el pipeline de main.c ---
    
    # Etapa 1: Transformada Espectral
    print("(1/4) Ejecutando Transformada Espectral...")
    spectral_out = model.spectral_analysis_transform(image_tensor) # Forma: (1, 512, 512, 8)
    # Dumps opcionales del espectral (pre y post normalización Lambda)
    dump_dir = "debug_dumps"
    os.makedirs(dump_dir, exist_ok=True)
    if os.environ.get("DUMP_SPECTRAL") == "1":
        spec_planar = tf.transpose(spectral_out, (0,3,1,2))  # (1,8,512,512)
        spec_planar = tf.squeeze(spec_planar, axis=0)        # (8,512,512)
        spec_planar.numpy().astype(np.float32).tofile(os.path.join(dump_dir, "spectral_py.bin"))
        # Normalización equivalente a la Lambda de Analysis (x/65535)
        spec_norm = spectral_out / tf.constant(((2**bit_length) - 1), dtype=tf.float32)
        specn_planar = tf.transpose(spec_norm, (0,3,1,2))
        specn_planar = tf.squeeze(specn_planar, axis=0)
        specn_planar.numpy().astype(np.float32).tofile(os.path.join(dump_dir, "spectral_norm_py.bin"))
        print("        [DEBUG] Espectral volcado a spectral_py.bin y spectral_norm_py.bin")
    
    # Etapa 2: Analysis Transform
    print("(2/4) Ejecutando Analysis Transform...")
    
    # Reordenar para el 'batch' de AnalysisTransform
    # (1, 512, 512, 8) -> (1, 8, 512, 512)
    x_1D = tf.transpose(spectral_out, (0, 3, 1, 2))
    # (1, 8, 512, 512) -> (8, 1, 512, 512)
    x_1D = tf.reshape(x_1D, (BANDS, 1, H_IN, W_IN))
    # (8, 1, 512, 512) -> (8, 512, 512, 1)
    x_1D = tf.transpose(x_1D, (0, 2, 3, 1))
    
    # Opción de volcado intermedio para diagnosticar conv0 y gdn0
    dump_stages = os.environ.get("DUMP_STAGES", "0") == "1"
    if dump_stages:
        # Accede a la primera capa conv+GDN de analysis
        l0 = model.analysis_transform.layers[1]  # SignalConv2D con GDN activation
        # Calcula conv0 pre-activación con tf.nn.conv2d (incluyendo la normalización Lambda: x/65535)
        k0 = l0.kernel
        b0 = l0.bias
        # tf.nn.conv2d es correlación por defecto; strides_down=2, padding SAME
        x_norm = x_1D / tf.constant(((2**bit_length) - 1), dtype=tf.float32)
        conv0_pre = tf.nn.conv2d(x_norm, k0, strides=[1, 2, 2, 1], padding="SAME")
        if b0 is not None:
            conv0_pre = tf.nn.bias_add(conv0_pre, b0)
        # Dump conv0_pre como planar (C,H,W) por banda agregada: (8,256,256,128) -> (3072,256,256)
        conv0_pre_planar = tf.transpose(conv0_pre, (0,3,1,2))  # (8,128,256,256)
        conv0_pre_planar = tf.reshape(conv0_pre_planar, (BANDS*C0, H1, W1))
        conv0_pre_planar.numpy().astype(np.float32).tofile(os.path.join(dump_dir, "conv0_pre_py.bin"))
        # Aplicar GDN de la capa (misma instancia/epsilon/pesos)
        gdn0_out = l0.activation(conv0_pre)
        gdn0_planar = tf.transpose(gdn0_out, (0,3,1,2))  # (8,128,256,256)
        gdn0_planar = tf.reshape(gdn0_planar, (BANDS*C0, H1, W1))
        gdn0_planar.numpy().astype(np.float32).tofile(os.path.join(dump_dir, "gdn0_py.bin"))
        # Cálculo manual de GDN usando bins exportados para validar fórmula/orientación
        try:
            gamma_bin = np.fromfile(os.path.join("weights", "pesos_bin", "analysis_gdn_0_gamma.bin"), dtype=np.float32).reshape((C0, C0))
            beta_bin = np.fromfile(os.path.join("weights", "pesos_bin", "analysis_gdn_0_beta.bin"), dtype=np.float32).reshape((C0,))
            eps = float(getattr(l0.activation, "epsilon", 1.0))
            # conv0_pre shape: (8, 256, 256, 128) -> reordenar a (8, 128, 256*256)
            c0_nchw = tf.transpose(conv0_pre, (0,3,1,2))  # (8,128,256,256)
            c0_flat = tf.reshape(c0_nchw, (BANDS, C0, H1*W1))
            x2 = tf.square(c0_flat).numpy()  # (8,128,65536)
            # denom[i, p] = beta[i] + sum_j gamma[i,j] * x2[j,p]
            denom = np.einsum('ij,bjk->bik', gamma_bin, x2)  # (8,128,65536)
            denom += beta_bin[None, :, None]
            denom = np.maximum(denom, eps)
            norm = np.sqrt(denom)
            y_manual = (c0_flat.numpy() / norm).astype(np.float32)
            # Volver a (8,128,256,256)
            y_manual_nchw = y_manual.reshape((BANDS, C0, H1, W1))
            y_manual_planar = y_manual_nchw.reshape((BANDS*C0, H1, W1))
            y_manual_planar.astype(np.float32).tofile(os.path.join(dump_dir, "gdn0_manual_py.bin"))
            # Comparar contra gdn0_planar
            ref = gdn0_planar.numpy().astype(np.float32).reshape((-1,))
            man = y_manual_planar.astype(np.float32).reshape((-1,))
            diff = np.abs(ref - man)
            print(f"        [DEBUG] GDN0 manual vs capa: max={diff.max():.6g} mean={diff.mean():.6g}")
            # Variante reparametrizada (hipótesis 1: parámetros al cuadrado)
            beta_sq = beta_bin * beta_bin
            gamma_sq = gamma_bin * gamma_bin
            denom_sq = np.einsum('ij,bjk->bik', gamma_sq, x2) + beta_sq[None, :, None]
            denom_sq = np.maximum(denom_sq, eps)
            y_sq = c0_flat.numpy() / np.sqrt(denom_sq)
            y_sq_planar = y_sq.reshape((BANDS*C0, H1, W1)).astype(np.float32)
            diff_sq = np.abs(ref - y_sq_planar.reshape(-1))
            print(f"        [DEBUG] GDN0 manual (beta,gamma)^2 vs capa: max={diff_sq.max():.6g} mean={diff_sq.mean():.6g}")
            # Variante reparametrizada (hipótesis 2: softplus)
            def softplus(x):
                return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
            beta_sp = softplus(beta_bin)
            gamma_sp = softplus(gamma_bin)
            denom_sp = np.einsum('ij,bjk->bik', gamma_sp, x2) + beta_sp[None, :, None]
            denom_sp = np.maximum(denom_sp, eps)
            y_sp = c0_flat.numpy() / np.sqrt(denom_sp)
            y_sp_planar = y_sp.reshape((BANDS*C0, H1, W1)).astype(np.float32)
            diff_sp = np.abs(ref - y_sp_planar.reshape(-1))
            print(f"        [DEBUG] GDN0 manual softplus(beta,gamma) vs capa: max={diff_sp.max():.6g} mean={diff_sp.mean():.6g}")
        except Exception as e:
            print(f"        [WARN] No se pudo calcular GDN0 manual: {e}")
        # --- Etapa siguiente: conv1 + gdn1 ---
        l1 = model.analysis_transform.layers[2]
        k1 = l1.kernel
        b1 = l1.bias
        conv1_pre = tf.nn.conv2d(gdn0_out, k1, strides=[1, 2, 2, 1], padding="SAME")
        if b1 is not None:
            conv1_pre = tf.nn.bias_add(conv1_pre, b1)
        gdn1_out = l1.activation(conv1_pre)
        # gdn1_out shape: (8,128,128,128); transpose then flatten bands*channels
        gdn1_planar = tf.transpose(gdn1_out, (0,3,1,2))
        gdn1_planar = tf.reshape(gdn1_planar, (BANDS*C0, H2, W2))  # C0==C1==128
        gdn1_planar.numpy().astype(np.float32).tofile(os.path.join(dump_dir, "gdn1_py.bin"))

        # --- Etapa siguiente: conv2 + gdn2 ---
        l2 = model.analysis_transform.layers[3]
        k2 = l2.kernel
        b2 = l2.bias
        conv2_pre = tf.nn.conv2d(gdn1_out, k2, strides=[1, 2, 2, 1], padding="SAME")
        if b2 is not None:
            conv2_pre = tf.nn.bias_add(conv2_pre, b2)
        gdn2_out = l2.activation(conv2_pre)
        gdn2_planar = tf.transpose(gdn2_out, (0,3,1,2))  # (8,128,64,64)
        gdn2_planar = tf.reshape(gdn2_planar, (BANDS*C0, H3, W3))  # C0==C2==128
        gdn2_planar.numpy().astype(np.float32).tofile(os.path.join(dump_dir, "gdn2_py.bin"))

        # --- Última capa conv3 (sin activación, sin bias) ---
        l3 = model.analysis_transform.layers[4]
        k3 = l3.kernel  # No hay bias
        conv3_pre = tf.nn.conv2d(gdn2_out, k3, strides=[1, 2, 2, 1], padding="SAME")  # (8,32,32,384)
        conv3_planar = tf.transpose(conv3_pre, (0,3,1,2))  # (8,384,32,32)
        conv3_planar = tf.reshape(conv3_planar, (BANDS*C3_AN, H4, W4))
        conv3_planar.numpy().astype(np.float32).tofile(os.path.join(dump_dir, "conv3_py.bin"))
        print("        [DEBUG] conv0_pre, gdn0, gdn1, gdn2 y conv3 volcados a *_py.bin")

    # Ejecutar la Analysis Transform por etapas (conv/GDN) para alinear EXACTAMENTE con la implementación en C
    # 1) Capa 0
    l0 = model.analysis_transform.layers[1]
    k0, b0 = l0.kernel, l0.bias
    x_norm = x_1D / tf.constant(((2**bit_length) - 1), dtype=tf.float32)
    conv0_pre = tf.nn.conv2d(x_norm, k0, strides=[1, 2, 2, 1], padding="SAME")
    if b0 is not None:
        conv0_pre = tf.nn.bias_add(conv0_pre, b0)
    gdn0_out = l0.activation(conv0_pre)
    # 2) Capa 1
    l1 = model.analysis_transform.layers[2]
    k1, b1 = l1.kernel, l1.bias
    conv1_pre = tf.nn.conv2d(gdn0_out, k1, strides=[1, 2, 2, 1], padding="SAME")
    if b1 is not None:
        conv1_pre = tf.nn.bias_add(conv1_pre, b1)
    gdn1_out = l1.activation(conv1_pre)
    # 3) Capa 2
    l2 = model.analysis_transform.layers[3]
    k2, b2 = l2.kernel, l2.bias
    conv2_pre = tf.nn.conv2d(gdn1_out, k2, strides=[1, 2, 2, 1], padding="SAME")
    if b2 is not None:
        conv2_pre = tf.nn.bias_add(conv2_pre, b2)
    gdn2_out = l2.activation(conv2_pre)
    # 4) Capa 3 (sin activación, sin bias)
    l3 = model.analysis_transform.layers[4]
    k3 = l3.kernel
    conv3_pre = tf.nn.conv2d(gdn2_out, k3, strides=[1, 2, 2, 1], padding="SAME")  # (8,32,32,384)

    # Concatenar como en main.c a (1, 32, 32, 3072)
    y = conv3_pre  # Usamos el resultado por etapas (equivale a Y_pre_c)
    y_3D = tf.transpose(y, (1, 2, 0, 3))
    # (32, 32, 8, 384) -> (1, 32, 32, 3072)
    y_3D = tf.reshape(y_3D, (1, H4, W4, BANDS * C3_AN))
    
    # Etapa 3: Modulating Transform
    # Permitimos replicar el escalado externo como en C usando la variable de entorno MAX_LAMBDA.
    # Si MAX_LAMBDA > 0, aplicamos lambda_scaled = lambda / MAX_LAMBDA ANTES de entrar al Sequential.
    # Si no, usamos lambda_tensor tal cual (el Sequential ya tendrá su propia capa Lambda interna si se definió así en el modelo guardado).
    max_lambda_env = os.environ.get("MAX_LAMBDA", "")
    lambda_input = lambda_tensor
    if max_lambda_env:
        try:
            max_lambda_val = float(max_lambda_env)
            if max_lambda_val > 0.0:
                print(f"(3/4) Ejecutando Modulating Transform (lambda/max_lambda: {LAMBDA_VAL}/{max_lambda_val})...")
                lambda_input = lambda_tensor / max_lambda_val
            else:
                print("(3/4) Ejecutando Modulating Transform (MAX_LAMBDA <= 0 ignorado)...")
        except ValueError:
            print("[WARN] MAX_LAMBDA no es convertible a float; usando lambda original.")
            print("(3/4) Ejecutando Modulating Transform...")
    else:
        print("(3/4) Ejecutando Modulating Transform...")
    modulator = model.modulating_transform(lambda_input) # Forma: (1, 1, 1, 3072)
    # Diagnóstico: inferir el maxval interno de la primera Lambda del ModulatingTransform.
    try:
        lambda_layer = model.modulating_transform.layers[0]
        probe_in = tf.constant([[[[1.0]]]], dtype=tf.float32)
        probe_out = lambda_layer(probe_in).numpy().ravel()[0]
        inferred_maxval = 1.0 / probe_out if probe_out != 0 else float('inf')
        print(f"        [DEBUG] Escalado interno ModulatingTransform: lambda=1 -> {probe_out:.6g}; maxval~{inferred_maxval:.6g}")
    except Exception as e:
        print(f"        [WARN] No se pudo inferir maxval interno: {e}")
    if os.environ.get("DUMP_M") == "1":
        m_np = np.squeeze(modulator.numpy())  # (3072,)
        m_np.astype(np.float32).tofile(os.path.join(dump_dir, "M_py.bin"))
        print("        [DEBUG] Modulator exportado a M_py.bin (3072 elementos)")

    # Dump opcional del tensor Y antes de la modulación para diagnóstico
    if os.environ.get("DUMP_Y_PRE") == "1":
        y_pre_np = tf.transpose(y_3D, (0, 3, 1, 2))  # (1, 3072, 32, 32)
        y_pre_np = tf.squeeze(y_pre_np, axis=0)      # (3072, 32, 32)
        y_pre_np.numpy().astype(np.float32).tofile(os.path.join(dump_dir, "Y_pre_py.bin"))
        print("        [DEBUG] Tensor Y antes de modulación exportado a Y_pre_py.bin")

    # Etapa 4: Cuantización
    print("(4/4) Aplicando Modulación y Cuantización...")
    y_float = y_3D * modulator
    if os.environ.get("DUMP_Y_FLOAT") == "1":
        yf = tf.transpose(y_float, (0,3,1,2))  # (1,3072,32,32)
        yf = tf.squeeze(yf, axis=0)
        yf.numpy().astype(np.float32).tofile(os.path.join(dump_dir, "Y_float_py.bin"))
        print("        [DEBUG] Producto Y*M exportado a Y_float_py.bin")
    y_hat = tf.round(y_float) # y_hat tiene forma (1, 32, 32, 3072)

    # Guardar el tensor
    # Lo guardamos como (C, H, W) (planar) para que coincida con la salida de C
    y_hat_planar = tf.transpose(y_hat, (0, 3, 1, 2)) # (1, 3072, 32, 32)
    y_hat_planar = tf.squeeze(y_hat_planar, axis=0) # (3072, 32, 32)
    
    output_path = os.path.join(dump_dir, OUTPUT_FILE)
    print(f"Guardando tensor 'Y_hat' en '{output_path}'...")
    y_hat_planar.numpy().astype(np.float32).tofile(output_path)
    
    print("\n--- Validación de Python Completada ---")
    print(f"Archivo 'verdad absoluta' generado: {output_path}")

if __name__ == "__main__":
    main()