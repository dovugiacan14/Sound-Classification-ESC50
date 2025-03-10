import logging
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


def display_model_comparision(results):
    """
    Display a comparison of model performance based on test accuracy.

    This function creates a DataFrame from the given results, sorts it by test accuracy in 
    descending order, applies a gradient-based styling, and visualizes the accuracy of 
    different models using a bar plot.

    Args:
        results (list of dict): A list containing model evaluation results with keys 
                                "Model" and "Test Accuracy".

    Returns:
        pd.DataFrame: A styled DataFrame showing sorted model accuracy.
    """
    df_results = pd.DataFrame(results).sort_values(by="Test Accuracy", ascending=False)
    df_results_styled = df_results.style.background_gradient(cmap="coolwarm")
    
    print(df_results_styled)
    
    sns.barplot(x="Test Accuracy", y="Model", data=df_results, palette="coolwarm")
    plt.xlabel("Accuracy (%)")
    plt.ylabel("Model")
    plt.title("Comparison of Model Accuracy")
    plt.show()


def display_accuracy_summary(summary_stats):
    df_summary = pd.DataFrame(summary_stats)
    logging.info("\n🔹 Accuracy Summary Table:")
    logging.info("\n%s", df_summary.to_string(index=False))

def display_model_results(model_results):
    for df in model_results:
        model_name = df["Model"][0]
        logging.info("\n🔹 Detailed Prediction Table - %s:", model_name)
        logging.info("\n%s", df[['File', 'True Label', 'Predicted Label', 'Correct']].head(10).to_string(index=False))

def plot_confusion_matrices(models_dict, X_test, y_test, random_indices):
    plt.figure(figsize=(15, 10))
    for i, (name, model) in enumerate(models_dict.items()):
        y_pred = np.argmax(model.predict(X_test.iloc[random_indices]), axis=1)
        y_true = y_test.iloc[random_indices].values

        plt.subplot(2, 3, i + 1)
        sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="coolwarm")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Confusion Matrix - {name}")
    
    plt.tight_layout()
    plt.show()
