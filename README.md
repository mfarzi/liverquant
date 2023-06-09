# liverquant
A python package for automated Whole Slide Image (WSI) analysis to quantitate fatty liver. The toolbox supports fat globule detection (see Examples 1 and 3) and fibrosis estimation (see Example 2). We plan to extend the toolbox to quantify inflammation and ballooning in future. 

## Istallation
The recommended way to install is via pip:

`pip install liverquant`

## Example 1: Detect Fat Globules
In liver pathology, the presence of fat can be indicative of various conditions, such as fatty liver disease (steatosis), which can occur in the context of alcohol abuse, obesity, diabetes, or metabolic syndrome. H&E staining (Hematoxylin and Eosin staining) is a widely used staining technique in pathology that provides basic information about tissue architecture and cellular morphology. Under H&E staining, lipids (fats) appear as clear or pale vacuoles within cells. The image tile used in this example is download from [histology page](https://gtexportal.org/home/histologyPage) with the tissue sample ID _GTEX-12584-1526_. 

To detect fat globules, use the `detect_fat_globules` function. The function first segment blobs in white. For color segmentation, you need to provide the lower and upper bounds for _Hue-Saturation-Value_ channels. Note that similar to _OpenCV_, Hue has values from 0 to 180, Saturation and Value from 0 to 255. To pick the white color, Saturation between 0 and 25 is filtered out by default. Overlapping fat globules are seperated using a watershed segmentation. Fat globules are then identified by filtering blobs based on morphological features including size, solidity, and elongation.

<figure>
  <img src="https://github.com/mfarzi/liverquant/raw/main/example/fat_detection.jpg" alt="Fat Detection Example" style="width:70%; margin-right:10px;" />
</figure>

```
import cv2 as cv
from liverquant import detect_fat_globules

# read sample image tile
img = cv.imread('./example/tile01_HE.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# Detect globules
mask = detect_fat_globules(img, resolution=0.4942)                                           

# Tag globules with green color                                   
img[mask == 255, :] = [5, 255, 5]

# write the tagged image
cv.imwrite('./example/tile01_HE_tagged.jpg', cv.cvtColor(img, cv.COLOR_BGR2RGB))
```

## Example 2: Collagen Segmentation
Picrosirius red (PSR) stain is a special staining technique used in liver pathology to evaluate collagen fibers. It is commonly employed to assess liver fibrosis, a condition characterized by the excessive accumulation of collagen in the liver due to chronic liver diseases such as hepatitis, alcoholic liver disease, and non-alcoholic fatty liver disease. The PSR stain selectively binds to collagen fibers, enabling their visualization under a microscope. Under a microscope, the PSR stain highlights collagen fibers as distinct red or orange-red structures. An image patch stained with PSR is provided in the example folder. 

To segment the collagen shown in red, use the `segment_by_color` function. You need to provide the lower and upper bounds for _Hue-Saturation-Value_ channels. Note that similar to _OpenCV_, Hue has values from 0 to 180, Saturation and Value from 0 to 255. To pick the red color, Hue between 0 and 10 or 170-180 should be filtered out. We have combined the two ranges into one effective range [-10, 10] in the sample code below. Since Hue is inherently periodic, negative Hue can be interpreted as positive integers by adding 180.

<figure>
  <img src="https://github.com/mfarzi/liverquant/raw/main/example/collagen_segmentation.jpg" alt="Collage Segmentation Example" style="width:80%; margin-right:10px;" />
</figure>

```
import cv2 as cv
from liverquant import segment_by_color

# read sample image tile
img = cv.imread('./example/tile02_PSR.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# Segment collagen using Hue-Saturation-Value channels
# To fill in holes, either set hole_size=None or increase the hole_size
mask = segment_by_color(img,
                        lowerb=[-10, 50, 100],
                        upperb=[10, 255, 255],
                        resolution=1.0124,
                        hole_size=25)

# write the tagged image
cv.imwrite('./example/tile02_PSR_mask.jpg', mask)
```

## Example 3: Detect Fat Globules in the Whole Slide Image
To identify fat globules in the whole slide image, utilize the `detect_fat_globules_wsi` function. This function divides the input image into overlapping image tiles and applies the `detect_fat_globules` algorithm (refer to Example 1) to each tile. It is important to note that `detect_fat_globules_wsi` employs parallel computation, and the entire script should be enclosed within a `if __name__ == '__main__'` block. Liverquant export detected geometrical features in [geojson](https://geojson.org/) format using the [`cv2geojson`](https://github.com/mfarzi/cv2geojson) python package, which can be visualised using dedicated software tools like [QuPath](https://qupath.github.io/). The WSI used in this example is not provided due to its large size but can be downloaded from [histology page](https://gtexportal.org/home/histologyPage) with the tissue sample ID _GTEX-12584-1526_. The estimated fat proportionate area is 12.8% and the run time was about 224 seconds.

<figure>
  <img src="https://github.com/mfarzi/liverquant/raw/main/example/fat_detection_wsi_qupath_screenshot.png" alt="Visualise Fat globules in WSI using QuPath" style="width:80%; margin-right:10px;" />
</figure>

```
from liverquant import detect_fat_globules_wsi
from cv2geojson import export_annotations
from openslide import OpenSlide

if __name__ == '__main__':
    # open the whole slide image
    slide = OpenSlide('./example/GTEX-12584-1526.svs')

    # detect fat globules
    fat_proportionate_area, globules, roi, run_time = detect_fat_globules_wsi(slide, cores_num=12)

    # export geojson features
    features = []
    for geocontour in roi:
        features.append(geocontour.export_feature(color=(0, 0, 255), label='foreground'))
    for geocontour in globules:
        features.append(geocontour.export_feature(color=(0, 255, 0), label='fat'))
    export_annotations(features, './example/GTEX-12584-1526.geojson')

    # print out results
    print(f'The fat proportionate area is {fat_proportionate_area}%.')
    print(f'The run time is {run_time} seconds.')
```