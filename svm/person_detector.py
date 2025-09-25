import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("persons_pics_train.csv")
print("data size:", data.shape)

unique_persons= data["label"].unique()
print("unique persons:", unique_persons, unique_persons.shape)
data_persons = data.groupby("label")
freq_data = data_persons.size()
print("freq person data:", freq_data)
'''sns.barplot(data=freq_data)
plt.show()'''

mean_data = data_persons.mean()
print("mean data:", mean_data)
print(mean_data.loc["Gerhard Schroeder", "0"])

'''
for person in unique_persons:
    plt.imshow(mean_data.loc[person].astype(float).to_numpy().reshape(62,47), cmap='gray')
    plt.axis('off')
    plt.title(person)
    plt.show()
'''

gs = mean_data.loc["Gerhard Schroeder"]
hc = mean_data.loc["Hugo Chavez"]
cos_similarity = (gs @ hc) / (np.linalg.norm(gs)*np.linalg.norm(hc))
print("cos similarity of mean Gerhard Schroeder and Hugo Chavez", cos_similarity)

