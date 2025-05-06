import os
import time
import numpy as np
import tensorflow as tf
import mlflow
from mlflow.exceptions import MlflowException
from model import create_model
from pathlib import Path

# Создаем healthcheck файл при старте
HEALTHCHECK_FILE = Path("/app/healthcheck")
HEALTHCHECK_FILE.touch()

def create_health_file():
    """Create health check file"""
    with open("/tmp/healthy", "w") as f:
        f.write("ready")

def remove_health_file():
    """Remove health check file if exists"""
    if os.path.exists("/tmp/healthy"):
        os.remove("/tmp/healthy")

def wait_for_mlflow(max_attempts=30, delay=2):
    """Wait for MLflow server to become available"""
    for attempt in range(max_attempts):
        try:
            mlflow.set_tracking_uri("http://mlflow-server:5000")
            mlflow.get_experiment_by_name("CIFAR10 Cats vs Dogs")
            return True
        except (MlflowException, Exception) as e:
            print(f"Attempt {attempt + 1}/{max_attempts}: MLflow server not ready - {str(e)}")
            time.sleep(delay)
    return False

def filter_cats_dogs(x, y):
    """Filter only cats (class 3) and dogs (class 5) from CIFAR10"""
    mask = np.isin(y.flatten(), [3, 5])
    x_filtered = x[mask]
    y_filtered = y[mask]
    y_filtered = np.where(y_filtered == 3, 0, 1)  # cats=0, dogs=1
    return x_filtered, y_filtered

def prepare_data():
    """Load and prepare CIFAR10 data"""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, y_train = filter_cats_dogs(x_train, y_train)
    x_test, y_test = filter_cats_dogs(x_test, y_test)
    
    # Нормализуем значения пикселей
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    return (x_train, y_train), (x_test, y_test)

def train():
    # Для создания файла проверки работоспособности
    create_health_file()
    
    try:
        # Ждем старта сервера
        if not wait_for_mlflow():
            raise RuntimeError("Failed to connect to MLflow server after multiple attempts")
        
        # Подготовка данных
        (x_train, y_train), (x_test, y_test) = prepare_data()
        
        # MLflow setup
        mlflow.set_experiment("CIFAR10 Cats vs Dogs")
        
        with mlflow.start_run():
            # Параметры логгирования
            params = {
                "batch_size": 64,
                "epochs": 10,
                "learning_rate": 0.001,
                "optimizer": "adam"
            }
            mlflow.log_params(params)
            
            # Создание и компиляция модели
            model = create_model()
            model.compile(
                optimizer=tf.keras.optimizers.Adam(params["learning_rate"]),
                loss='binary_crossentropy',
                metrics=['accuracy',
                        tf.keras.metrics.Precision(name='precision'),
                        tf.keras.metrics.Recall(name='recall')]
            )
            
            # Тренировка модели
            print("Starting training...")
            history = model.fit(
                x_train, y_train,
                batch_size=params["batch_size"],
                epochs=params["epochs"],
                validation_data=(x_test, y_test),
                verbose=1
            )
            
            # Метрики
            mlflow.log_metrics({
                "final_train_accuracy": history.history['accuracy'][-1],
                "final_val_accuracy": history.history['val_accuracy'][-1],
                "final_train_loss": history.history['loss'][-1],
                "final_val_loss": history.history['val_loss'][-1]
            })
            
            mlflow.keras.log_model(model, "model")
            print("Training completed successfully!")
            
    except Exception as e:
        print(f"Training failed: {str(e)}")
        remove_health_file()
        raise
    finally:
        if os.path.exists("/tmp/healthy"):
            os.remove("/tmp/healthy")

if __name__ == "__main__":
    print("Starting training script...")
    try:
        train()
    except Exception as e:
        print(f"Fatal error in training: {str(e)}")
        exit(1)