import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

data = pd.read_csv("94_16.csv")

pca = PCA(svd_solver="full")
pca.fit(data)

#print(pca.explained_variance_ratio_)

#print(pca.singular_values_)

pca_basis_data = pca.transform(data)
'''m = data.mean()
_m_matrix = [np.full((data.shape[0], 1), x) for x in m]
m_matrix = np.concatenate(_m_matrix, axis=1)'''

centrilized_data = data-data.mean()

print(pca_basis_data[0], centrilized_data.iloc[0] @ pca.components_[1])