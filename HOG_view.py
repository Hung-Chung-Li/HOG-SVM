# -*- coding: utf-8 -*-
from skimage import exposure

from skimage import feature

import cv2

imagePath = "train/rock/rock07-k03-084.png"

image = cv2.imread(imagePath)

image = cv2.resize(image, (500, 500))

cv2.imshow("Origin", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

(H, hogImage) = feature.hog(gray, orientations=9, pixels_per_cell=(8, 8),

    cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1",

    visualize=True)

hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
hogImage = hogImage.astype("uint8")

cv2.imshow("Scikit-image", hogImage)

cv2.waitKey(0)