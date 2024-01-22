from pathlib import Path
import os

from matching_algorithm.nlp_utils import preprocess_text


def load_tokens(data_dir: Path):
    all_tokens = []
    for fn in os.listdir(data_dir):
        with open(data_dir / fn , 'r') as file:
            # Read all lines into a list
            interest_text = file.readlines()
        interest_list = preprocess_text(interest_text)
        all_tokens.append(interest_list)
    return all_tokens
