"""Data preprocessing stage: reads raw CSV and writes processed output."""
import os
import yaml
import pandas as pd


def preprocess(input_path, output_path):
    """Read raw dataset and write processed CSV.

    Args:
        input_path: Path to raw CSV file.
        output_path: Path to write processed CSV.
    """
    df = pd.read_csv(input_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, header=None, index=False)

    print(f'Preprocessed data saved to {output_path}')


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)['preprocess']

    preprocess(params['input'], params['output'])