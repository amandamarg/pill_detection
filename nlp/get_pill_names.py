import os
import pandas as pd
from config.nlp_paths_selector import nlp_configs


def main():
    leaflet_files = pd.Series(os.listdir(nlp_configs().get("patient_information_leaflet_doc")))
    pill_names = leaflet_files.map(lambda x: x.replace('_', ' ').split('.')[0]).sort_values()
    path = nlp_configs().get('pill_names') + '/pill_names.xlsx'
    pill_names.to_excel(path, index=None, header=None)

if __name__ == "__main__":
    main()

