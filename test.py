import cv2 as cv
from liverquant import detect_fat_globules

if __name__ == '__main__':
    # read sample image tile
    img = cv.imread('./example/tile01_HE.jpg')
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Detect globules
    mask = detect_fat_globules(img, resolution=0.4942)

    # Tag globules with green color
    img[mask == 255, :] = [5, 255, 5]

    # write the tagged image
    cv.imwrite('./example/tile01_HE_tagged.jpg', cv.cvtColor(img, cv.COLOR_BGR2RGB))
