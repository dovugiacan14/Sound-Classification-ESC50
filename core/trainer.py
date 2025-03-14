import logging

def train(model_name, model, X_train, X_test, y_train, y_test):
    logging.info(f"\n🔵 Training {model_name}...")
    try:
        history = model.fit(
            X_train,
            y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1,
        )
        train_acc = history.history["accuracy"][-1]
        train_loss = history.history["loss"][-1]
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        results = {
            "Model": model_name,
            "Train Accuracy": train_acc,
            "Train Loss": train_loss,
            "Test Accuracy": test_acc,
            "Test Loss": test_loss,
        }
        return results 
    
    except Exception as e:
        logging.error(f"Error occurred when training model: {model_name}: {e}")
