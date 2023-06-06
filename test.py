import cv2 as cv
from liverquant import detect_fat_globules
from matplotlib import pyplot as plt
# read sample image
img = cv.imread('./images/GTEX-12584-1526-patch.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# Detect globules
mask, _ = detect_fat_globules(img,
                              lowerb=[0, 0, 100],
                              upperb=[180, 30, 255],
                              resolution=0.4942,
                              hole_max=20)

# Tag globules with green color
img[mask == 255, :] = [5, 255, 5]
