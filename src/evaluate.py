import pandas as pd
import yaml
import json
import joblib
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_params(config_path):
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)
    return params['evaluate'],params['preprocess'],params['train']


def main():
    evaluate_config, preprocess_config, train_config = load_params('params.yaml')

    test_csv_path = preprocess_config['output_test']
    model_path = train_config['model_output']

    test_df = pd.read_csv(test_csv_path)
    model = joblib.load(model_path)

    X_test = test_df.drop('churn', axis=1)
    y_test = test_df['churn']

    y_pred_test = model.predict(X_test)

    metrics = {
        "test_accuracy": accuracy_score(y_test, y_pred_test),
        "test_precision": precision_score(y_test, y_pred_test),
        "test_recall": recall_score(y_test, y_pred_test),
        "test_f1": f1_score(y_test, y_pred_test)
    }

    metrics_output_path = evaluate_config['metrics_output']
    with open(metrics_output_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    try:
        if mlflow.active_run() is not None:
            mlflow.log_metrics(metrics)
    except Exception:
        print("MLFlow run not active")

if __name__ == "__main__":
    main()