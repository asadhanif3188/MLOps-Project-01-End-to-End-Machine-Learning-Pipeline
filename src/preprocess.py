"""Data preprocessing stage: reads raw CSV and writes processed output."""
from dotenv import load_dotenv

from logging_config import configure_logging, get_logger
from pipeline_io import load_params, read_csv, write_csv
from stage_runner import run_stage

logger = get_logger("preprocess")


def preprocess(input_path: str, output_path: str) -> None:
    """Read raw dataset and write processed CSV.

    Args:
        input_path: Path to raw CSV file.
        output_path: Path to write processed CSV.

    Raises:
        DataError: If the input cannot be read or the output cannot be written.
    """
    logger.info("Preprocess stage started (input=%s, output=%s)", input_path, output_path)

    df = read_csv(input_path)
    write_csv(df, output_path, header=False, index=False)

    logger.info("Preprocess stage completed: %d rows written to %s", len(df), output_path)


def main() -> None:
    """Entry point: load environment, configure logging, run the stage."""
    load_dotenv()
    configure_logging()

    params = load_params("params.yaml", "preprocess", required=("input", "output"))
    preprocess(params["input"], params["output"])


if __name__ == "__main__":
    run_stage("preprocess", main)
