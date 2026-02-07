import pandas as pd
import yaml

params = yaml.safe_load(open("params.yaml"))

def load_data():
    df = pd.read_csv(params["data"]["raw_path"])
    return df

if __name__ == "__main__":
    df = load_data()
    print(df.head())