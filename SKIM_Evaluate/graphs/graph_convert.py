import pandas as pd
import numpy as np
import pickle

df = pd.read_excel("graph_false.xlsx")


arrays = [df[col].to_numpy() for col in df.columns]


graph_array = np.array(arrays, dtype=object)  # 用 object 防止维度对齐错误


with open("graph_false.pkl", "wb") as f:
    pickle.dump(graph_array, f)
