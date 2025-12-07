import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

def load_params(config_path):
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)
    return params['preprocess'], params['generate']


def main():
    preprocess_config, generate_config = load_params('params.yaml')

    raw_csv_path = generate_config['output_path']
    df = pd.read_csv(raw_csv_path)

    df = df.drop('customer_id', axis=1)

    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    test_size = preprocess_config['test_size']
    random_state = preprocess_config['random_state']


    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['churn']
    )

    output_train = preprocess_config['output_train']
    output_test = preprocess_config['output_test']

    train_df.to_csv(output_train, index=False)
    test_df.to_csv(output_test, index=False)

    print(f'Processed train: {output_train}')
    print(f'Processed train: {output_test}')

if __name__ == '__main__':
    main()

