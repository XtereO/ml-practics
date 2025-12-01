import sys

from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import cv2
import numpy as np

def mean_intensity(img):
    return np.mean(img)

# Путь к изображению
img_path = 'spb_bridge.jpg'

img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(img)

if len(sys.argv)>1 and sys.argv[1]=="1":
    plt.show()

print("avg intensity pixels:", mean_intensity(img))

img_scaled = img/255
print("avg intensity scaled pixels:", mean_intensity(img_scaled))

h, w, c = img.shape
img_scaled_flat = img_scaled.reshape((h*w, 3))

# so we can store for each pixel only 4 bits! 
model = MiniBatchKMeans(n_clusters=16, random_state=18).fit(img_scaled_flat)

clusters = model.cluster_centers_
compressed_img = np.empty((h*w, c))
for i, p in enumerate(model.labels_):
    compressed_img[i] = clusters[p]

print("avg of clustered pixels:", mean_intensity(compressed_img))