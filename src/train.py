import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def load_params(config_path):
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)
    return params['train'], params['preprocess']

def main():
    train_config, preprocess_config = load_params('params.yaml')

    mlflow.set_experiment("customer_churn_dvc_mlflow")

    with mlflow.start_run():
        train_csv_path = preprocess_config['output_train']
        train_df = pd.read_csv(train_csv_path)

        X_train = train_df.drop('churn', axis=1)
        y_train = train_df['churn']

        model_type = train_config['model_type']

        if model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=train_config['n_estimators'],
                max_depth=train_config['max_depth'],
                random_state=train_config['random_state']
            )

            mlflow.log_params({
                "model_type": model_type,
                "n_estimators": train_config['n_estimators'],
                "max_depth": train_config['max_depth']
            })
        else:
            print("Unknown model", model_type)


        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_pred_train)

        mlflow.log_metric("training_acc", train_accuracy)
        print(f"Traning Acc: {train_accuracy:.4f}")

        model_output_path = train_config['model_output']
        joblib.dump(model, model_output_path)

        mlflow.sklearn.log_model(model, "model")
        print("Trained model")

if __name__ == "__main__":
    main()
