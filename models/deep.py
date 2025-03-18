import tensorflow as tf
from keras import layers, models

class NNModel:
    def __init__(self, input_shape, num_classes):
        """
        Initialize the model with input shape and number of classes.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_cnn_lstm(self):
        """
        Build a CNN + LSTM model for feature extraction and sequence learning.
        """
        inputs = layers.Input(shape=self.input_shape)

        # CNN for spatial feature extraction
        x = layers.Reshape((self.input_shape[0], 1))(inputs)
        x = layers.Conv1D(64, kernel_size=3, activation="relu", padding="same")(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Conv1D(128, kernel_size=3, activation="relu", padding="same")(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Conv1D(256, kernel_size=3, activation="relu", padding="same")(x)
        x = layers.MaxPooling1D(pool_size=2)(x)

        # LSTM for sequential learning
        x = layers.LSTM(128, return_sequences=True, dropout=0.3)(x)
        x = layers.LSTM(64, return_sequences=False, dropout=0.3)(x)

        # Fully Connected Layer
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.4)(x)

        outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        model = models.Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        return model

    def build_bilstm(self):
        """
        Build a BiLSTM + CNN model for bidirectional sequential learning.
        """
        inputs = layers.Input(shape=self.input_shape)

        # CNN for feature extraction
        x = layers.Reshape((self.input_shape[0], 1))(inputs)
        x = layers.Conv1D(256, kernel_size=5, padding="same", activation="relu")(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
        x = layers.MaxPooling1D(pool_size=2)(x)

        # BiLSTM for context learning
        x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.3))(x)
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.3))(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=False, dropout=0.3))(x)

        # Fully Connected Layer
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.4)(x)

        outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        model = models.Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        return model

    def build_cnn(self):
        """
        Build a simple CNN model.
        """
        model = models.Sequential([
            layers.Dense(512, activation="relu", input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation="softmax")
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        return model

    def attention_layer(self, inputs):
        """
        Implement Attention mechanism to focus on important features.
        """
        attention_weights = layers.Dense(1, activation="tanh")(inputs)
        attention_weights = layers.Flatten()(attention_weights)
        attention_weights = layers.Activation("softmax")(attention_weights)
        attention_weights = layers.RepeatVector(inputs.shape[-1])(attention_weights)
        attention_weights = layers.Permute([2, 1])(attention_weights)

        return layers.Multiply()([inputs, attention_weights])

    def build_lstm(self):
        """
        Build a BiLSTM model with Attention mechanism.
        """
        inputs = layers.Input(shape=self.input_shape)

        # CNN for feature extraction
        x = layers.Reshape((self.input_shape[0], 1))(inputs)
        x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
        x = layers.MaxPooling1D(pool_size=2)(x)

        # BiLSTM for sequential learning
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)

        # Attention Mechanism
        x = self.attention_layer(x)

        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.3)(x)

        outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        model = models.Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        return model
    
