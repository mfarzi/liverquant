# liverquant
A python package for automated Whole Slide Image (WSI) analysis to quantitate fatty liver. The toolbox supports fat globule detection (see [Example 1](#example-1) and [Example 2](#example-2)) and fibrosis estimation (see [Example 3](#example-3)). We plan to extend the toolbox to quantify inflammation and ballooning in future. 

## Contents
- [Installation](#installation)
- [Fat Qauntification](#fat-quantification)
  - [detect_fat_globules](#detec_fat_globules)
  - [detect_fat_globules_wsi](#detect_fat_globules_wsi)
- [Fibrosis Qauntification](#fibrosis-quantification)
  - [segment_by_color](#segment_by_color)

## Istallation
The recommended way to install is via pip:

`pip install liverquant`

## Fat Qauntification
In liver pathology, the presence of fat can be indicative of various conditions, such as fatty liver disease (steatosis), which can occur in the context of alcohol abuse, obesity, diabetes, or metabolic syndrome. Hematoxylin and Eosin (H&E) staining is a widely used technique in pathology that provides basic information about tissue architecture and cellular morphology. Under H&E staining, lipids (fats) appear as clear or pale vacuoles within cells. The workflow to identify fat vauoles is shown below. In brief, white regions are segmented using _Hue-Saturation-Value_ channels. Isolated fat vacuoles are segmented using morphological features. Overlapping fat globules are seperated using a watershed segmentation and then identified using the same filters applied to isolated fat vacuoles.

<figure>
  <img src="https://github.com/mfarzi/liverquant/raw/main/example/fat_quant_workflow.jpg" alt="Fat Quantification Workflow" style="width:90%; margin-right:10px;" />
</figure>

> Note: The image tile used in this example is download from [histology page](https://gtexportal.org/home/histologyPage) with the tissue sample ID _GTEX-12584-1526_. 

The libarary implements three main methods to identify fat globules.

### <code>detect_fat_globules</code>
Retreive a binary mask for fat vacoules from input RGB image tile.
- Parameters:
  - img: {numpy.ndarray}: input RGB image tile in range [0, 255]
  - mask=None: {numpy.ndarray}: binary mask (either 0 or 255) for regions-of-interest
  - lowerb=None: {list: 3}: inclusive lower bound array in HSV-space for color segmentation
  - upperb=None: {list: 3}: inclusive upper bound array in HSV-space for color segmentation
  - overalp=0: {int}: globules with centre inside overlap region will be excluded to avoid double-counting in whole-slide-image
  - resolution=1.0: {float}: pixel size in micro-meter
  - min_diameter=5.0: {float}: minimum diameter of a white blob to be considered as a fat globule
  - max_diameter=100.0: {float}: maximum diameter of a white blob to be considered as a fat globule
- Returns:
  - mask: {numpy.ndarray}: binary mask (either 0 or 255) for identified fat vacuoles

> For color segmentation, you need to provide the lower and upper bounds for _Hue-Saturation-Value_ channels. Note that similar to _OpenCV_, Hue has values from 0 to 180, Saturation and Value from 0 to 255. To pick the white color, Saturation between 0 and 25 is filtered out by default. Overlapping fat globules are seperated using a watershed segmentation. Fat globules are then identified by filtering blobs based on morphological features including size, solidity, and elongation.

#### Example 1
Here is a short script to demonstrate the utility of `detect_fat_globules`.
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

### <code>detect_fat_globules_wsi</code>
Retreive fat vacoules in `cv2geojson.GeoContour` format from the whole slide image (WSI).
- Parameters:
  - slide: {openslide}: A handle object to the whole slide image
  - roi: {list: cv2geojson.GeoContours}: regions-of-interest in the WSI frame [default=None]
  - lowerb: {list: 3}: inclusive lower bound array in HSV-space for color segmentation [default=[0, 0, 200]]
  - upperb: {list: 3}: inclusive upper bound array in HSV-space for color segmentation [default=[180, 25, 255]]
  - tile_size: {int}: the tile size to sweep the whole slide image [default=2048]
  - overalp: {int}: globules with centre inside overlap region will be excluded to avoid double-counting in whole-slide-image [default=128]
  - downsample: {int}: downsampling ratio as a power of two [default=2]
  - min_diameter: {float}: minimum diameter of a white blob to be considered as a fat globule [default=5]
  - max_diameter: {float}: maximum diameter of a white blob to be considered as a fat globule [default=100]
  - cores_num: {int}: max number of cores to be used for parallel computation 
- Returns:
  - fat_proportion_area: {float}: the amount of fat in the tissue represented in percent
  - vacoules: {list: cv2geojson.GeoContour}: polygons representing fat vacoules
  - roi: {list: cv2geojson.GeoContour}: same as input roi if provided; otherwise, representing the foreground mask
  - run_time: {float}: run time for the code completion in seconds

> This function divides the input image into overlapping image tiles and applies the `detect_fat_globules` algorithm (refer to [Example 1](#example-1)) to each image tile. `detect_fat_globules_wsi` employs parallel computation, and the entire script should be enclosed within a `if __name__ == '__main__'` block. Liverquant export detected geometrical features in [geojson](https://geojson.org/) format using the [`cv2geojson`](https://github.com/mfarzi/cv2geojson) python package, which can be visualised using dedicated software tools like [QuPath](https://qupath.github.io/). 

#### Example 2
Here is a short script to demonstrate the utility of `detect_fat_globules_wsi`. The WSI used in this example is not provided due to its large size but can be downloaded from [histology page](https://gtexportal.org/home/histologyPage) with the tissue sample ID _GTEX-12584-1526_. The estimated fat proportionate area is 12.8% and the run time was about 224 seconds.

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

## Fibrosis Qauntification
Liver fibrosis is a progressive condition characterized by the excessive accumulation of extracellular matrix components, particularly collagen, within the liver tissue. It is typically a consequence of chronic liver injury caused by various factors such as viral hepatitis, alcohol abuse, metabolic disorders, or autoimmune diseases. The excessive collagen deposition disrupts the normal liver architecture and impairs its function over time.

To evaluate the extent of liver fibrosis, histological staining techniques are commonly employed. Picrosirius red (PSR), Masson's Trichrome (MTC), and Van Gieson (VG) staining are three widely used methods to visualize and quantify the collagen content within liver tissue. These techniques allow for the identification of collagen fibers and provide valuable information about the severity and distribution of fibrosis.

1- Picrosirius red (PSR) staining: PSR staining specifically highlights collagen fibers and helps differentiate between mature (thick, red-orange staining) and immature (thin, green staining) collagen. Under polarized light, the stained fibers exhibit birefringence, which provides additional information about their structural organization. This staining technique allows for the assessment of collagen distribution, fiber thickness, and the presence of newly formed collagen.

2- Masson's Trichrome (MTC) staining: MTC staining is a widely used technique to visualize collagen fibers in tissues. It involves staining collagen fibers blue or green, nuclei dark blue or black, and cytoplasm and muscle fibers red. Collagen fibers appear as blue-green structures, allowing for the identification and quantification of fibrosis. MTC staining provides information about the extent of collagen deposition and can help evaluate fibrosis severity.

3- Van Gieson (VG) staining: VG staining is another method used to assess collagen deposition in tissues. It stains collagen fibers red or pink, while other tissue components such as muscle fibers, cytoplasm, and nuclei are stained yellow or brown. This staining technique allows for the identification and quantification of collagen within the liver tissue, enabling the evaluation of fibrosis severity.

The libarary implements two main methods to identify collagens using colour segmentation in both individual image tiles and the whole slide image.

### <code>segment_by_color</code>
Retreive a binary mask for segmented regions based on their colour profile in the HSV space.
- Parameters:
  - img: {numpy.ndarray}: input RGB image tile in range [0, 255]
  - mask=None: {numpy.ndarray}: binary mask (either 0 or 255) for regions-of-interest
  - lowerb: {list: 3}: inclusive lower bound array in HSV-space for color segmentation [default=[0, 0, 200]]
  - upperb: {list: 3}: inclusive upper bound array in HSV-space for color segmentation [default=[0, 25, 255]]
  - hole_size: {float}: remove holes smaller than hole_size; if zero, all holes will be reserved. If -1, all holes
                        will be removed. [default=0]
  - resolution: {float}: pixel size in micro-meter [default=1]
  
- Returns:
  - mask: {numpy.ndarray}: binary mask (either 0 or 255) for segmented regions

> Note that similar to _OpenCV_, Hue has values from 0 to 180, Saturation and Value from 0 to 255.

#### Example 3
Here is a short script to demonstrate the utility of `segment_by_color`. An image patch stained with PSR is provided in the example folder. To segment the collagen shown in red, Hue between 0 and 10 or 170-180 should be filtered out. We have combined the two ranges into one effective range [-10, 10] in the sample code below. Since Hue is inherently periodic, negative Hue can be interpreted as positive integers by adding 180. Note that similar to _OpenCV_, Hue has values from 0 to 180.

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

