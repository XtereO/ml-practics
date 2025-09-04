import pandas as pd
import numpy as np

def enorm(x, xmin):
    return (1-np.exp(1-x/xmin))
  

data = pd.read_csv("asset-v1_ITMOUniversity+DATST+spring_2023_ITMO_bac+type@asset+block@task4_1_777064.csv")
np_data = data.to_numpy()

data_min = (data.drop("ID", axis=1).min())
print(data_min)
data_norm = data
data_norm["DISTANCE"] = data_norm["DISTANCE"].astype(float)
data_norm["FITNESS"] = data_norm["FITNESS"].astype(float)
data_norm["COST"] = data_norm["COST"].astype(float)
data_norm["STOP_COUNT"] = data_norm["STOP_COUNT"].astype(float)

for i in range(200): 
    data_norm.loc[i, "DISTANCE"] = enorm(data_norm.loc[i, "DISTANCE"], data_min["DISTANCE"]) 
    data_norm.loc[i, "STOP_COUNT"] = enorm(data_norm.loc[i, "STOP_COUNT"], data_min["STOP_COUNT"])
    data_norm.loc[i, "FITNESS"] = enorm(data_norm.loc[i, "FITNESS"], data_min["FITNESS"])
    data_norm.loc[i, "COST"] = enorm(data_norm.loc[i, "COST"], data_min["COST"])

min_sum=4.0
best_id=0
for i in range(200):
    s = (data_norm.loc[i, "DISTANCE"]+data_norm.loc[i, "STOP_COUNT"]+data_norm.loc[i, "FITNESS"]+data_norm.loc[i, "COST"])
    if min_sum > s:
        best_id = data_norm.loc[i, "ID"]
        min_sum = s

print(best_id)
     