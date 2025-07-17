import os
import pandas as pd
import numpy as np

if __name__=="__main__":
    root_path = "../output/17_07_2025"
    paths = os.listdir(root_path)
    print(f'paths : {paths}')
    results = []
    for dir_name in paths:
        result_path = f"{root_path}/{dir_name}/result_metrics_{dir_name}.csv"
        result_df = pd.read_csv(result_path)
        results.append(result_df)

    numeric_dfs = [df.drop(columns=['Model Name']) for df in results]

    mean_array = np.mean([df.values for df in numeric_dfs], axis=0)
    mean_df = pd.DataFrame(mean_array, columns=numeric_dfs[0].columns)

    mean_df.insert(0, 'Model Name', results[0]['Model Name'].values)

    mean_df.to_csv(f"{root_path}/model_results.csv", index=False)
    print(mean_df)
