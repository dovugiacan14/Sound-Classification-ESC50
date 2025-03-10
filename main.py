import os
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import *
from core.trainer import train
from data.argument import DataArgumentor
from data.extraction import extract_audio_features
from data.preprocessing import split_and_normalize
from core.models import AudioCNN
from core.predict import evaluate_models
from visualizer.visualize import (
    display_model_comparision, 
    display_accuracy_summary, 
    display_model_results, 
    plot_confusion_matrices, 
)

data_path = DATA_PATH, 
meta_data_path = META_DATA_PATH
audio_path = AUDIO_PATH 
augmented_path = AUGMENTED_PATH

def main():
    # Load dataset
    csv_path = os.path.join(data_path, meta_data_path)
    dataset = pd.read_csv(csv_path)
    dataset["files_path"] = dataset["filename"].apply(
        lambda x: os.path.join(audio_path, x)
    )

    # Encode label
    encoder = LabelEncoder()
    dataset["target"] = encoder.fit_transform(dataset["category"])
    num_classes = len(encoder.classes_)

    # Augment data
    data_augmentor = DataArgumentor(dataset, augmented_path)
    augmented_files, augmented_targets = data_augmentor.process()
    dataset_aug = pd.DataFrame(
        {"files_path": augmented_files, "target": augmented_targets}
    )
    dataset = pd.concat([dataset, dataset_aug], ignore_index=True)
    logging.info(f"\n🟢 Total samples after augmenting: {len(dataset)}")

    # Extract audio features
    dataset, features = extract_audio_features(dataset)

    # Train-test split
    X_train, X_test, y_train, y_test = split_and_normalize(dataset, features)

    # Train models
    audio_cnn_classifier = AudioCNN(
        input_shape=(X_train.shape[1],), num_classes=num_classes
    )

    cnn_model = audio_cnn_classifier.build_cnn()
    cnn_trained_results = train(
        model_name= "CNN", 
        model= cnn_model, 
        X_train= X_train,
        X_test= X_test, 
        y_train= y_train, 
        y_test= y_test
    )

    bilstm_model = audio_cnn_classifier.build_bilstm()
    bilstm_trained_results = train(
        model_name= "Bi-LSTM", 
        model= bilstm_model , 
        X_train= X_train,
        X_test= X_test, 
        y_train= y_train, 
        y_test= y_test
    )

    transformers_model = audio_cnn_classifier.build_transformer()
    transformers_trained_results = train(
        model_name= "Transformers", 
        model= transformers_model, 
        X_train= X_train,
        X_test= X_test, 
        y_train= y_train, 
        y_test= y_test
    )
    
    trained_results = {
        "CNN": cnn_trained_results,
        "Bi-LSTM": bilstm_trained_results,
        "Transformers": transformers_trained_results,
    }
    display_model_comparision(trained_results) 

    # evaluate model 
    model_results, summary_stats = evaluate_models(
        models_dict= trained_results, 
        X_test= X_test, 
        y_test= y_test, 
        dataset= dataset, 
        encoder= encoder, 
        random_indices= np.random.choice(len(X_test), 1000, replace=False)
    )

    display_accuracy_summary(summary_stats)
    display_model_results(model_results)

    plot_confusion_matrices(
        models_dict= trained_results, 
        X_test= X_test, 
        y_test= y_test, 
        random_indices= np.random.choice(len(X_test), 1000, replace=False)
    )

if __name__ == "__main__":
    main()
