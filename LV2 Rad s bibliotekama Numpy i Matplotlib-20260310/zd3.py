import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("road.jpg")
img = img.astype(float) / 255

img_light = img + 0.2
img_light = np.clip(img_light, 0, 1)
plt.imshow(img_light)
plt.show()

_, w, _ = img.shape
druga_cetvrtina = img[:, w//4:w//2]
plt.imshow(druga_cetvrtina)
plt.show()

rotirana = np.rot90(img, -1)
plt.imshow(rotirana)
plt.show()

zrcaljena = np.fliplr(img)
plt.imshow(zrcaljena)
plt.show()