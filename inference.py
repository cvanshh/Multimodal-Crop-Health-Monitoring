"""Model loading, prediction, Grad-CAM, and a clearly-labelled heuristic fallback."""
import numpy as np

from config import (CNN_PATH, LSTM_PATH, FUSION_PATH, CLASSES_PATH, REFSPEC_PATH, META_PATH, RESULTS_PATH,
                    DEFAULT_CLASS_NAMES, IMG_SIZE)
from data_pipeline import load_json


class Predictor:
    def __init__(self):
        self.model = self.lstm = self.fusion = None
        self.error = None
        self.class_names = load_json(CLASSES_PATH, DEFAULT_CLASS_NAMES)
        self.reference = load_json(REFSPEC_PATH, None)
        self.results = load_json(RESULTS_PATH, {}) or {}
        self.metrics = self.results.get("cnn")
        _meta = load_json(META_PATH, {}) or {}
        self.img_size = _meta.get("img_size", IMG_SIZE)
        self.num_bands = _meta.get("num_bands")
        self._grad_model = None
        try:
            from tensorflow import keras
            if CNN_PATH.exists():
                self.model = keras.models.load_model(CNN_PATH, compile=False)
            if LSTM_PATH.exists():
                self.lstm = keras.models.load_model(LSTM_PATH, compile=False)
            if FUSION_PATH.exists():
                self.fusion = keras.models.load_model(FUSION_PATH, compile=False)
        except Exception as e:  # noqa: BLE001
            self.error = str(e)

    def sensor_status(self, window):
        """LSTM P(healthy) for a (10,5) scaled window, or None."""
        if self.lstm is None or window is None:
            return None
        return float(self.lstm.predict(window[None].astype(np.float32), verbose=0)[0, 0])

    def fusion_predict(self, cube_norm, window):
        if self.fusion is None or window is None:
            return None
        return self.fusion.predict([cube_norm[None].astype(np.float32), window[None].astype(np.float32)], verbose=0)[0]

    @property
    def available(self):
        return self.model is not None

    def predict(self, cube_norm):
        """cube_norm: (H,W,B) in [0,1] -> probability vector."""
        return self.model.predict(cube_norm[None].astype(np.float32), verbose=0)[0]

    def gradcam(self, cube_norm, class_idx=None):
        import tensorflow as tf
        from tensorflow import keras
        if self._grad_model is None:
            self._grad_model = keras.Model(
                self.model.inputs,
                [self.model.get_layer("last_conv").output, self.model.output])
        x = tf.convert_to_tensor(cube_norm[None], tf.float32)
        with tf.GradientTape() as tape:
            conv, preds = self._grad_model(x, training=False)
            if class_idx is None:
                class_idx = int(tf.argmax(preds[0]))
            score = preds[:, class_idx]
        grads = tape.gradient(score, conv)
        if grads is None:
            return np.zeros(conv.shape[1:3], np.float32)
        w = tf.reduce_mean(grads, axis=(1, 2))
        cam = tf.nn.relu(tf.reduce_sum(conv * w[:, None, None, :], axis=-1))[0].numpy()
        return (cam / cam.max()).astype(np.float32) if cam.max() > 1e-8 else cam.astype(np.float32)


def heuristic_probs(ndvi_mean, class_names):
    """Rule-of-thumb stand-in used ONLY when no trained model exists. Not a real classifier."""
    z = {"health": (ndvi_mean - 0.45) * 8, "other": 1.0 - abs(ndvi_mean - 0.45) * 8,
         "rust": (0.45 - ndvi_mean) * 8}
    logits = []
    for c in class_names:
        key = "health" if c.lower().startswith("health") else "rust" if "rust" in c.lower() else "other"
        logits.append(z[key])
    e = np.exp(np.array(logits) - max(logits))
    return e / e.sum()
