import os
import pandas as pd
import yaml
import numpy as np

def load_params(config_path):
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)
    return params['generate']

def main():
    config = load_params('params.yaml')
    n_samples = config['n_samples']
    churn_rate = config['churn_rate']
    output_path = config['output_path']

    np.random.seed(42)

    data = {}
    data['customer_id'] = [f'C{i:04d}' for i in range(n_samples)]
    data['tenure_months'] = np.random.randint(1,73,n_samples)
    data['contract_type'] = np.random.choice(['month-to-month', 'one-year', 'two-year'], n_samples, p=[0.5, 0.3, 0.2])
    data['monthly_charges'] = np.random.uniform(20,120,n_samples)
    data['has_internet_service'] = np.random.choice(['dsl', 'fiber', 'none'], n_samples, p=[0.4,0.4,0.2])
    data['has_tech_support'] = np.random.randint(0,2,n_samples)

    df = pd.DataFrame(data)


    p_churn = np.full(n_samples, churn_rate)

    p_churn[df['contract_type']=='month-to-month'] *= 1.5
    p_churn[df['has_internet_service'] == 'fiber'] *= 1.4
    p_churn[df['tenure_months'] < 12] *= 1.3
    p_churn[df['monthly_charges'] > 80] *= 1.2

    p_churn[df['tenure_months'] >48] *= 0.5
    p_churn[df['contract_type'] == 'two-year'] *= 0.3
    p_churn[df['has_tech_support'] == 1] *= 0.7

    p_churn = np.clip(p_churn, 0.05, 0.95)

    df['churn'] = (np.random.rand(n_samples) < p_churn).astype(int)

    df.to_csv(output_path, index=False)
    print(f"Generated churn data -> {output_path}")

if __name__ == "__main__":
    main()
