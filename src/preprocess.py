import pandas
import sys
import yaml
import os


def preprocess(input_path, output_path):
    """
    Preprocess the data by reading the input file, 
    dropping the 'Unnamed: 0' column,
    and saving the preprocessed data to a CSV file.
    """
    # Read the input file
    df = pandas.read_csv(input_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the preprocessed data to a CSV file
    df.to_csv(output_path, header=None, index=False)

    print(f'Preprocessed data saved to {output_path}')

if __name__ == "__main__":
    # Load parameters from param.yaml
    params = yaml.safe_load(open("params.yaml"))['preprocess']

    # Call the preprocess function with the input and output paths
    preprocess(
        params['input'], 
        params['output']
        )