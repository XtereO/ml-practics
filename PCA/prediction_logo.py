import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

X_loading = pd.read_csv("X_loadings_588.csv", sep=";")
X_reduced = pd.read_csv("X_reduced_588.csv", sep=";")

# transforming to initial basis so we can get an image that was at the start
# with shifted coordinates (cause avg dif in PCA)
plt.imshow(X_reduced.to_numpy() @ (X_loading.to_numpy()).T, cmap='gray')
plt.show()
