import keras
import tensorflow as tf
from keras import layers, models


class AudioCNN:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def attention_layer(self, inputs):
        """
        Attention mechanism that helps the model focus on important features.

        This layer applies a dense transformation followed by a softmax activation
        to generate attention weights. The input features are then weighted accordingly
        before applying global average pooling.

        Args:
            inputs (tf.Tensor): The input tensor to apply attention on.

        Returns:
            tf.Tensor: The output tensor after applying attention and global average pooling.
        """
        attention_weights = layers.Dense(64, activation="tanh")(inputs)
        attention_weights = layers.Dense(1, activation="softmax")(attention_weights)
        attention_weights = layers.Multiply()([inputs, attention_weights])

        return layers.GlobalAveragePooling1D()(attention_weights)

    def build_cnn(self):
        model = models.Sequential(
            [
                layers.Dense(512, activation="relu", input_shape=self.input_shape),
                layers.BatchNormalization(),
                layers.Dense(256, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(128, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(self.num_classes, activation="softmax"),
            ]
        )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def build_bilstm(self):
        inputs = layers.Input(shape=self.input_shape)

        # add CNN layer to learn features
        x = layers.Reshape((self.input_shape[0], 1))(inputs)
        x = layers.Conv1D(256, kernel_size=5, padding="same", activation="relu")(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
        x = layers.MaxPooling1D(pool_size=2)(x)

        # Bi-directional LSTM
        x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.3))(
            x
        )
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.3))(
            x
        )
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.3))(x)

        # add Attention to focus important feature
        x = self.attention_layer(x)

        # add BatchNormalization for stable training
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.4)(x)

        outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        model = models.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0003),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def build_transformer(self):
        inputs = layers.Input(shape=self.input_shape)

        # convert input from 2D to 3D
        x = layers.Reshape((self.input_shape[0], 1))(inputs)

        # Positional Encoding
        x = layers.Conv1D(
            filters=128,
            kernel_size=3,
            padding="same",
            activation="relu",
        )(x)
        x = layers.LayerNormalization()(x)

        # Multi-head Attention with many layers
        for _ in range(3):
            attn_output = layers.MultiHeadAttention(num_heads=16, key_dim=64)(x, x)
            x = layers.Add()([x, attn_output])
            x = layers.LayerNormalization()(x)

        x = layers.GlobalAveragePooling1D()(x)
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        model = models.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model
