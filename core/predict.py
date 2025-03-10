import logging
import numpy as np
import pandas as pd


def evaluate_models(models_dict, X_test, y_test, dataset, encoder, random_indices):
    """
    Evaluate multiple models on a test dataset and generate summary statistics.
    
    Args:
        models_dict (dict): Dictionary of models to evaluate.
        X_test (DataFrame): Test feature dataset.
        y_test (Series): True labels for the test dataset.
        dataset (DataFrame): Original dataset containing file paths.
        encoder (LabelEncoder): Encoder used to transform labels.
        random_indices (list): List of random indices for evaluation.

    Returns:
        tuple: (model_results, summary_stats)
            - model_results (list of DataFrame): Prediction results for each model.
            - summary_stats (list of dict): Summary statistics for each model.
    """
    model_results = []
    summary_stats = []
    
    for name, model in models_dict.items():
        logging.info(f"\n🟢 Evaluating model: {name}")
        y_pred = np.argmax(model.predict(X_test.iloc[random_indices]), axis=1)
        y_true = y_test.iloc[random_indices].values

        # ✅ Convert labels back to actual category names
        true_labels = encoder.inverse_transform(y_true)
        predicted_labels = encoder.inverse_transform(y_pred)

        # ✅ Aggregate prediction results
        results = [
            {
                "Model": name,
                "File": dataset.iloc[idx]["files_path"],
                "True Label": true_labels[i],
                "Predicted Label": predicted_labels[i],
                "Correct": "✅" if true_labels[i] == predicted_labels[i] else "❌"
            }
            for i, idx in enumerate(random_indices)
        ]

        df_results = pd.DataFrame(results)

        # ✅ Compute correct/incorrect predictions
        correct_predictions = sum(y_true == y_pred)
        incorrect_predictions = len(y_true) - correct_predictions
        accuracy = round((correct_predictions / len(y_true)) * 100, 2)

        # ✅ Store summary statistics
        summary_stats.append({
            "Model": name,
            "Correct Predictions": correct_predictions,
            "Incorrect Predictions": incorrect_predictions,
            "Accuracy (%)": accuracy
        })

        # ✅ Save prediction results for each model
        model_results.append(df_results)
    
    return model_results, summary_stats
