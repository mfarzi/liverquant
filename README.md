# liverquant
A python package for automated Whole Slide Image (WSI) analysis to quantitate fatty liver. The toolbox supports fat globule detection (see Example 1) and fibrosis estimation (see Example 2). We plan to extend the toolbox to quantify inflammation and ballooning in future. 

## Introduction
In digital pathology, images are often quite large and dedicated software tools like [QuPath](https://qupath.github.io/) are required to aid visualisation.

## Example 1: Detect Fat Globules in an Image Tile
The image used in this example is download from [histology page](https://gtexportal.org/home/histologyPage). The tissue sample ID is _GTEX-12584-1526_. An image patch from this sample is included in the forlder images for testing purposes.

<div style="display:flex; justify-content:center; align-items:center; flex-direction:column;">
  <div style="display:flex; justify-content:center; align-items:center;">
    <figure style="margin:0; padding:0;">
      <img src="https://github.com/mfarzi/liverquant/raw/main/images/GTEX-12584-1526-patch.jpg" alt="Image 1" style="width:30%; margin-right:10px;" />
      <figcaption>Raw H&E Image Tile</figcaption>
    </figure>
    <figure style="margin:0; padding:0;">
      <img src="https://github.com/mfarzi/liverquant/raw/main/images/GTEX-12584-1526-tagged.jpg?raw=true" alt="Image 2" style="width:30%; margin-left:10px;" />
      <figcaption>Tagge H&E Image Tile with Detected Fat Globules in Green</figcaption>
    </figure>
  </div>
</div>

```
import cv2 as cv
from liverquant import detect_fat_globules

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

# write the tagged image
cv.imwrite('./images/GTEX-12584-1526-tagged.jpg', cv.cvtColor(img, cv.COLOR_BGR2RGB))
```

