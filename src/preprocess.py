"""Data preprocessing stage: reads raw CSV and writes processed output."""
import os

import pandas as pd
import yaml

from logging_config import configure_logging, get_logger

logger = get_logger("preprocess")


def preprocess(input_path, output_path):
    """Read raw dataset and write processed CSV.

    Args:
        input_path: Path to raw CSV file.
        output_path: Path to write processed CSV.
    """
    logger.info("Preprocess stage started (input=%s, output=%s)", input_path, output_path)

    df = pd.read_csv(input_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, header=None, index=False)

    logger.info("Preprocess stage completed: %d rows written to %s", len(df), output_path)


if __name__ == "__main__":
    configure_logging()

    with open("params.yaml") as f:
        params = yaml.safe_load(f)['preprocess']

    preprocess(params['input'], params['output'])