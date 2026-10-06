import os
import json
import argparse
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

# Si hay un servidor de tracking remoto configurado (Dagshub via variable de entorno), lo usamos
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)


def train_model(n_estimators):
    # Cargar el dataset versionado con DVC; si no existe, usar el dataset de sklearn como respaldo
    try:
        data = pd.read_csv("data/iris_dataset.csv")
        X = data.drop(columns=["target"])
        y = data["target"]
    except FileNotFoundError:
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target

    with mlflow.start_run():
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        joblib.dump(model, "model.pkl")

        mlflow.sklearn.log_model(model, "random-forest-model", skops_trusted_types=["sklearn.tree._tree.Tree"])
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("accuracy", accuracy)

        # Matriz de confusion para el informe de CML
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Prediccion")
        plt.ylabel("Real")
        plt.title("Matriz de confusion")
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")

        # Metricas para el reporte de CML
        with open("mlflow_metrics.json", "w") as f:
            json.dump({"accuracy": accuracy}, f)

        print(f"Modelo entrenado y precision: {accuracy:.4f}")
        print("Experimento registrado con MLflow.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    args = parser.parse_args()
    train_model(args.n_estimators)
