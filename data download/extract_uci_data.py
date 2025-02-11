import os

import pandas as pd
from ucimlrepo import fetch_ucirepo, list_available_datasets

list_available_datasets()

def save_dataframe(id):
    try:
        print(id)
        dataset = fetch_ucirepo(id=id)
        print(dataset.metadata['name'])
        if dataset.data['ids'] is not None and dataset.data['headers'] is not None:
            df = pd.DataFrame(dataset.data['original'])
        elif dataset.data['ids'] is None and dataset.data['headers'] is not None:
            df = pd.DataFrame(dataset.data['original'], columns=dataset.data['headers'])
        else:
            df = pd.DataFrame(dataset.data['original'], columns=dataset.data['headers'], index=dataset.data['ids'])
        saving_path = f"data_uci/{str(dataset.metadata['name']).replace(' ', '_')}"
        if not os.path.exists(saving_path):
            os.makedirs(saving_path)
        df.to_csv(f"{saving_path}/data.csv")
        dataset.variables.to_csv(f"{saving_path}/variables.csv")

    except:
        print("No dataset found")
        return
    return saving_path
