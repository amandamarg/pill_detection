import os
import pandas as pd
from config.nlp_paths_selector import nlp_configs


def main():
    full_sentences = pd.read_csv(nlp_configs().get("full_sentence_csv") + "/extracted_sentences.csv", header=None)
    path = nlp_configs().get('pill_names') + '/pill_names.xlsx'
    full_sentences[0].to_excel(excel_writer=path, index=None, header=None)

if __name__ == "__main__":
    main()

