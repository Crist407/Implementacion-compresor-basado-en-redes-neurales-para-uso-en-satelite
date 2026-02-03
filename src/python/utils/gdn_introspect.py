import tensorflow as tf
import tensorflow_compression as tfc

MODEL_DIR = "SORTENY_Sentinel2_model"

m = tf.keras.models.load_model(MODEL_DIR, compile=False)
layer = m.analysis_transform.layers[1]
gdn = layer.activation
print("GDN class:", gdn.__class__.__name__)
print("Attrs:", [a for a in dir(gdn) if not a.startswith('_')])
print("epsilon:", getattr(gdn, 'epsilon', None))
print("beta var shape:", getattr(gdn, 'beta', None).shape)
print("gamma var shape:", getattr(gdn, 'gamma', None).shape)
# Try to find reparam attributes
for name in ['beta_min', 'beta_bound', 'gamma_bound', 'reparam_offset', 'inverse', 'apply_renorm', 'data_format', 'rectify']:
    print(name, ":", getattr(gdn, name, None))

# Try compute effective beta/gamma if available
try:
    beta = gdn.beta
    gamma = gdn.gamma
    # Some GDNs use 'rectify' to ensure positivity via square or softplus
    rectify = getattr(gdn, 'rectify', None)
    if callable(rectify):
        beff = rectify(beta)
        geff = rectify(gamma)
        print("rectify present -> beff min/max:", tf.reduce_min(beff).numpy(), tf.reduce_max(beff).numpy())
        print("rectify present -> geff min/max:", tf.reduce_min(geff).numpy(), tf.reduce_max(geff).numpy())
except Exception as e:
    print("rectify check failed:", e)
