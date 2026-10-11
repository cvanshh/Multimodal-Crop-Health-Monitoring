"""Paper architecture: CNN (hyperspectral) + LSTM (IoT) branches with late fusion.

CNN : 3 blocks [Conv3x3-ReLU-BN-MaxPool] 32/64/128 -> Flatten -> Dense(256) -> Dropout(0.5) -> softmax(3)
LSTM: LSTM(64, seq) -> Dropout(0.2) -> LSTM(32) -> sigmoid(1)      (input 10 x 5)
Fusion: concat(256 + 32 = 288) -> Dense(64, ReLU) -> Dropout(0.3) -> softmax(3)
"""
from tensorflow import keras
from tensorflow.keras import layers


def build_cnn(input_shape, num_classes, lr=1e-3):
    inp = keras.Input(shape=input_shape, name="cube")
    x = inp
    for i, f in enumerate((32, 64, 128)):
        x = layers.Conv2D(f, 3, padding="same", activation="relu",
                          name="last_conv" if i == 2 else f"conv_{i}")(x)   # Grad-CAM layer
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(x)
    emb = layers.Dense(256, activation="relu", name="cnn_embedding")(x)
    x = layers.Dropout(0.5)(emb)
    out = layers.Dense(num_classes, activation="softmax", name="cnn_out")(x)
    m = keras.Model(inp, out, name="HSI_CNN")
    m.compile(keras.optimizers.Adam(lr), "sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def build_lstm(timesteps=10, features=5, lr=1e-3):
    inp = keras.Input(shape=(timesteps, features), name="sensors")
    x = layers.LSTM(64, return_sequences=True)(inp)
    x = layers.Dropout(0.2)(x)
    emb = layers.LSTM(32, name="lstm_embedding")(x)
    out = layers.Dense(1, activation="sigmoid", name="lstm_out")(emb)
    m = keras.Model(inp, out, name="IoT_LSTM")
    m.compile(keras.optimizers.Adam(lr), "binary_crossentropy", metrics=["accuracy"])
    return m


def build_fusion(cnn, lstm, num_classes, lr=1e-3):
    """Late fusion. Encoders are frozen (each modality learned independently); only the head trains."""
    enc_c = keras.Model(cnn.input, cnn.get_layer("cnn_embedding").output, name="cnn_encoder")
    enc_l = keras.Model(lstm.input, lstm.get_layer("lstm_embedding").output, name="lstm_encoder")
    enc_c.trainable = False
    enc_l.trainable = False
    img = keras.Input(shape=cnn.input_shape[1:], name="fusion_cube")
    seq = keras.Input(shape=lstm.input_shape[1:], name="fusion_sensors")
    z = layers.Concatenate()([enc_c(img, training=False), enc_l(seq, training=False)])   # 288-d
    z = layers.Dense(64, activation="relu")(z)
    z = layers.Dropout(0.3)(z)
    out = layers.Dense(num_classes, activation="softmax", name="fusion_out")(z)
    m = keras.Model([img, seq], out, name="CNN_LSTM_Fusion")
    m.compile(keras.optimizers.Adam(lr), "sparse_categorical_crossentropy", metrics=["accuracy"])
    return m
