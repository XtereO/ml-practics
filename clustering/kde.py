import numpy as np
import sys

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

digits = load_digits()
print("digits shape", digits.data.shape)

pca = PCA(n_components=0.99, svd_solver="full").fit(digits.data)
reduced_digits = pca.transform(digits.data)
print("reduced digits shape", reduced_digits.shape)

min_params = [50, float("Inf")]
gm_params = {"covariance_type": "full", "random_state": 109}
for i in range(min_params[0], 250, 10):
    gm = GaussianMixture(n_components=i, **gm_params).fit(reduced_digits)
    aic = gm.aic(reduced_digits)
    if min_params[1] > aic:
        min_params = [i, aic]
print("min params for GaussianMixture (n, aic):", min_params)

gm = GaussianMixture(
    n_components=min_params[0], **gm_params).fit(reduced_digits)
if (gm.converged_):
    print("converged successfully")
sample = gm.sample(100)[0]
i = 0
print(f"sample[:, {i}] mean",np.mean(sample[:, i]))

if len(sys.argv)>1 and sys.argv[1]=="1":
    sample_images = pca.inverse_transform(sample)
    fig, ax = plt.subplots(10, 10, figsize=(8, 8), subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for i, axi in enumerate(ax.flat):
        im = axi.imshow(sample_images[i].reshape(8,8), cmap='binary')
        im.set_clim(0, 16)
    plt.show()