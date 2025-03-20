import tensorflow as tf
from keras import layers, models

class NNModel:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def attention_layer(self, inputs):
        """
        Attention mechanism to focus on important features.
        """
        attention_weights = layers.Dense(1, activation="tanh")(inputs)
        attention_weights = layers.Flatten()(attention_weights)
        attention_weights = layers.Activation("softmax")(attention_weights)
        attention_weights = layers.RepeatVector(inputs.shape[-1])(attention_weights)
        attention_weights = layers.Permute([2, 1])(attention_weights)
        
        return layers.Multiply()([inputs, attention_weights])
    
    def build_cnn(self):
        """
        CNN model for feature extraction and classification.
        """
        model = models.Sequential([
            layers.Reshape((self.input_shape[0], 1), input_shape=self.input_shape),
            layers.Conv1D(128, kernel_size=3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(256, kernel_size=3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(512, kernel_size=3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation="softmax")
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model
    
    def build_lstm(self):
        """
        LSTM model with Conv1D for feature extraction before sequence learning.
        """
        inputs = layers.Input(shape=self.input_shape)
        x = layers.Reshape((self.input_shape[0], 1))(inputs)
        x = layers.Conv1D(128, kernel_size=3, activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        
        x = layers.Conv1D(256, kernel_size=3, activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        
        x = layers.LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)(x)
        x = layers.BatchNormalization()(x)
        
        x = self.attention_layer(x)
        
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)
        
        model = models.Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model
    
    def build_cnn_lstm(self):
        """
        Hybrid CNN-LSTM model combining feature extraction and sequence learning.
        """
        inputs = layers.Input(shape=self.input_shape)
        x = layers.Reshape((self.input_shape[0], 1))(inputs)
        x = layers.Conv1D(128, kernel_size=3, activation="relu", padding="same")(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Conv1D(256, kernel_size=3, activation="relu", padding="same")(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Conv1D(512, kernel_size=3, activation="relu", padding="same")(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        
        x = layers.LSTM(256, return_sequences=True, dropout=0.3)(x)
        x = layers.LSTM(128, return_sequences=True, dropout=0.3)(x)
        x = layers.LSTM(64, return_sequences=False, dropout=0.3)(x)
        
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)
        
        model = models.Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model
    
    def build_bilstm(self):
        """
        BiLSTM model for sequence classification.
        """
        inputs = layers.Input(shape=self.input_shape)
        x = layers.Reshape((self.input_shape[0], 1))(inputs)
        x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.3))(x)
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.3))(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=False, dropout=0.3))(x)
        
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)
        
        model = models.Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model