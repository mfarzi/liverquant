import numpy as np
import cv2 as cv
from cv2geojson import find_geocontours, draw_geocontours, export_annotations
from functools import wraps, partial
from .slidepatch import PatchGenerator, get_tile_image
from multiprocessing import Pool
import time
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


def get_contours_decorator(get_mask):
    @wraps(get_mask)
    def wrapper(*args, **kwargs):
        mask = get_mask(*args, **kwargs)
        contours = find_geocontours(mask, mode='imagej')
        return contours

    return wrapper


def count_pixels_decorator(get_mask):
    @wraps(get_mask)
    def wrapper(*args, **kwargs):
        mask = get_mask(*args, **kwargs)
        pixels_num = cv.countNonZero(mask)
        return pixels_num

    return wrapper


def get_mask_decorator(get_contours):
    @wraps(get_contours)
    def wrapper(*args, **kwargs):
        frame_size = args[0].shape[:2]
        geocontours = get_contours(*args, **kwargs)
        # draw mask
        mask = np.zeros(frame_size, dtype=np.uint8)
        draw_geocontours(mask, geocontours, mode='imagej')
        return mask

    return wrapper


def detect_fat_globules_contour(img, mask=None, lowerb=None, upperb=None, overlap=0, resolution=1.0, min_diameter=5.0,
                                max_diameter=100.0):
    """
    Detect fat globules by segmenting white blobs in the HSV space and then classify them using morphological features

    :param img:          A numpy array of input image tile
    :param mask:         A numpy array of selected ROI [either 255 or 0] corresponding to img
    :param lowerb:       inclusive lower bound array in HSV-space for color segmentation
    :param upperb:       inclusive upper bound array in HSV-space for color segmentation
    :param overlap:      Integer value; globules with centre inside overlap region will be excluded
    :param resolution:   pixel resolution in micron
    :param min_diameter: Minimum diameter of a white blob to be considered as a fat globule
    :param max_diameter: Maximum diameter of a white blob to be considered as a fat globule

    :return globules:   Binary mask of detected globules formatted as a numpy array [either 255 or 0]
    :return mask_white: Binary mask of detected white regions formatted as a numpy array [either 255 or 0]
    """
    if lowerb is None:
        lowerb = [0, 0, 200]
    if upperb is None:
        upperb = [180, 25, 255]

    # step 0: Detect all white blobs in the foreground
    mask_white = segment_by_color(img,
                                  mask=mask,
                                  lowerb=lowerb,
                                  upperb=upperb,
                                  hole_size=-1,
                                  resolution=resolution)

    # morphological opening using circular mask to remove spurious branches
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    mask_white = cv.morphologyEx(mask_white, cv.MORPH_OPEN, kernel=kernel, iterations=3)

    # step 1: Find non-overlapping fat globules
    # Extract geocontours for morphological feature extraction and globules detection
    geocontours = find_geocontours(mask_white, mode='imagej')
    # morphological filters
    globules, unknown, _ = filter_fat_globules(geocontours,
                                               x_range=(overlap, img.shape[0]-overlap),
                                               y_range=(overlap, img.shape[1]-overlap),
                                               solidity=(0.7, 0.85),
                                               elongation=(0.05, 0.4),
                                               diameter=(min_diameter, max_diameter),
                                               resolution=resolution)

    # step 2: Use watershed segmentation to separate overlapping globules
    mask = np.zeros_like(mask_white, dtype=np.uint8)
    draw_geocontours(mask, unknown, mode='imagej')
    geocontours = separate_globules(mask)

    # step 3: check if any seperated globules is fat
    globules_combined, _, _ = filter_fat_globules(geocontours,
                                                  x_range=(overlap, img.shape[0] - overlap),
                                                  y_range=(overlap, img.shape[1] - overlap),
                                                  solidity=(0.7, 0.85),
                                                  elongation=(0.05, 0.4),
                                                  diameter=(min_diameter, max_diameter),
                                                  resolution=resolution)
    globules.extend(globules_combined)
    return globules


@get_mask_decorator
def detect_fat_globules(*args, **kwargs):
    # wrapper function for detect_fat_globules
    result = detect_fat_globules_contour(*args, **kwargs)
    return result


def detect_fat_globules_wsi(slide, roi=None, lowerb=None, upperb=None, tile_size=2048, overlap=128, downsample=2,
                            min_diameter=5, max_diameter=100, cores_num=4):
    """
    Detect fat globules by sweeping over the whole slide image (WSI) tile by tile and return annotations (geojson)

    :param slide:                A openslide object to the whole slide image
    :param roi:                  A list of cv2geojson.GeoContours for the selected ROIs
    :param lowerb:               Inclusive lower bound array in HSV-space for color segmentation
    :param upperb:               Inclusive upper bound array in HSV-space for color segmentation
    :param tile_size:            The tile size to sweep the whole slide image
    :param overlap:              Integer value; globules with centre inside overlap region will be excluded
    :param downsample:           Downsampling ratio as a power of two
    :param min_diameter:         Minimum diameter of a white blob to be considered as a fat globule
    :param max_diameter:         Maximum diameter of a white blob to be considered as a fat globule
    :param cores_num:            Number of cores for parallel computation

    :return geocontours:         List of cv2geojson.GeoContour geometries representing fat globules as Polygons
    :return roi:                 list of cv2geojson.GeoContour geometries representing the foreground mask
    :return fat_proportion_area: The amount of fat in the tissue represented in percent
    :return run_time:            Run time for the code in seconds
    """
    start_time = time.time()
    if roi is None:
        roi = segment_foreground_wsi(slide)

    pixel_resolution = float(slide.properties['openslide.mpp-x'])

    # extract image tiles
    tiles = PatchGenerator(slide, tile_size=tile_size, overlap=overlap, downsample=downsample, roi=roi)
    print(f'Start parallel computation using {cores_num} cores...')
    with Pool(cores_num) as pool:
        partial_detect_fat_globules = partial(detect_fat_globules_contour,
                                              lowerb=lowerb,
                                              upperb=upperb,
                                              overlap=overlap,
                                              resolution=pixel_resolution,
                                              max_diameter=max_diameter,
                                              min_diameter=min_diameter)
        results = pool.starmap(partial_detect_fat_globules, tiles)
    pool.close()
    print('parallel coding is completed.')

    # pooling results and scaling contours
    geocontours = []
    for index, contours in enumerate(results):
        offset = tiles.address[index]
        # scale the contours
        for contour in contours:
            contour.scale_up(ratio=downsample, offset=offset)
        geocontours.extend(contours)

    # estimate fat proportin area
    area_tissue = np.sum([cnt.area(pixel_resolution) for cnt in roi])
    area_fat = np.sum([cnt.area(pixel_resolution) for cnt in geocontours])
    fat_proportion_area = np.round(area_fat / area_tissue * 100, 2)

    end_time = time.time()
    run_time = end_time - start_time
    return fat_proportion_area, geocontours, roi, run_time


def segment_by_color(img, mask=None, lowerb=None, upperb=None, hole_size=0.0, resolution=1.0):
    """
    Segment regions based on their color profile in HSV space

    :param img:          A numpy array of input image tile in RGB space
    :param mask:         A numpy array of desired ROIs [either 255 or 0] corresponding to img
    :param lowerb:       Inclusive lower bound array in HSV-space for color segmentation
    :param upperb:       Inclusive upper bound array in HSV-space for color segmentation
    :param hole_size:    Remove holes smaller than hole_size; if zero, all holes will be reserved. If -1, all holes
                         will be removed.
    :param resolution:   Image pixel resolution in micron
    :param mode:         Either 'opencv' or 'imagej'

    :return geocontours: A list of cv2geojson.GeoContour objects representing color segmented ROIs
    """
    # Default color is white
    if lowerb is None:
        lowerb = [0, 0, 200]
    if upperb is None:
        upperb = [180, 30, 255]

    # Convert image to HSV color space - Hue: 0-179, Saturation: 0-255, and Value: 0-255
    img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    if lowerb[0] < 0:
        # Handle negative Hue values by mapping the range 91-180 to -90-0
        h, s, v = cv.split(img_hsv)
        h = np.int32(h)
        h[h > 90] = h[h > 90] - 180
        mask_h = cv.inRange(h, lowerb[0], upperb[0])
        mask_s = cv.inRange(s, lowerb[1], upperb[1])
        mask_v = cv.inRange(v, lowerb[2], upperb[2])
        mask_color = cv.bitwise_and(cv.bitwise_and(mask_h, mask_s), mask_v)
    else:
        mask_color = cv.inRange(img_hsv, np.array(lowerb), np.array(upperb))

    # apply mask to only include regions inside the selected ROIs
    if mask is not None:
        mask_color = cv.bitwise_and(mask_color, mask)

    # remove small holes
    mask = fill_holes(mask_color, hole_size=hole_size, resolution=resolution)

    return mask


@get_contours_decorator
def segment_by_color_contour(*args, **kwargs):
    # wrapper function for segment_by_color_contour
    mask = segment_by_color(*args, **kwargs)
    return mask


@count_pixels_decorator
def count_pixels_by_color(*args, **kwargs):
    # wrapper function for segment_by_color
    result = segment_by_color(*args, **kwargs)
    return result


def _export_geocontour_parallel(file_name, geocontours, offset=(0, 0), scale=1):
    contours = [contour.scale_up(ratio=scale, offset=offset) for contour in geocontours]
    area_color = np.sum([cnt.area() for cnt in contours])
    features = [contour.export_feature() for contour in contours]
    export_annotations(features, file_name)
    return area_color


def segment_by_color_wsi(slide, roi=None, lowerb=None, upperb=None, tile_size=2048, downsample=2, hole_size=0,
                         cores_num=4):
    """
    Segment regions based on their color profile in HSV space by sweeping over the whole slide image (WSI) tile by tile
    and return annotations (geojson.Feature)

    :param slide:        An openslide handle object to the whole slide image
    :param roi:          A list of cv2geojson.GeoContours for the selected ROIs
    :param lowerb:       inclusive lower bound array in HSV-space for color segmentation
    :param upperb:       inclusive upper bound array in HSV-space for color segmentation
    :param tile_size:    The tile size to sweep the whole slide image
    :param downsample:   Downsampling ratio as a power of two
    :param hole_size:    Remove holes smaller than hole_size; if zero, all holes will be reserved. if -1, all holes
                         will be filled in [micro-meter squared]
    :param cores_num:    Number of cores for parallel computation
    :param mode:         Either 'imagej' or 'opencv'

    :return geocontours: List of cv2geojson.GeoContour geometries representing segmented ROIs as Polygons
    :return area_color:  total area of segmented ROIs [milli-meter-squared]
    :return run_time:    Run time for the code in seconds
    """
    if lowerb is None:
        lowerb = [0, 0, 200]
    if upperb is None:
        upperb = [180, 30, 255]

    pixel_resolution = float(slide.properties['openslide.mpp-x'])
    # pixel_resolution = 10000.0/float(slide.properties['tiff.XResolution'])

    # extract image tiles
    tiles = PatchGenerator(slide, tile_size=tile_size, overlap=0, downsample=downsample, roi=roi)
    print(f'Start parallel computation using {cores_num} cores...')
    start_time = time.time()
    with Pool(cores_num) as pool:
        partial_segment_by_color = partial(segment_by_color_contour,
                                           lowerb=lowerb,
                                           upperb=upperb,
                                           hole_size=hole_size,
                                           resolution=pixel_resolution * downsample)
        results = pool.starmap(partial_segment_by_color, tiles)
        # if output is not None:
        #     file_names = [str(output / 'tile_{}_{}.geojson'.format(loc[1], loc[0])) for loc in tiles.local_address]
        #     addresses = tiles.address
        #     partial_export_geocontour_parallel = partial(_export_geocontour_parallel, scale=downsample)
        #     area_color_list = pool.starmap(partial_export_geocontour_parallel, zip(file_names, results, addresses))
    pool.close()

    # pooling results and scaling contours
    geocontours = []

    for index, contours in enumerate(results):
        offset = tiles.address[index]
        # scale the contours
        for contour in contours:
            contour.scale_up(ratio=downsample, offset=offset)
        geocontours.extend(contours)
    area_color = np.sum([cnt.area() for cnt in geocontours]) * pixel_resolution * pixel_resolution * 1e-6

    end_time = time.time()
    run_time = end_time - start_time
    print(f'parallel coding is completed in {np.round(run_time, 3)} seconds')
    return geocontours, area_color, run_time


def count_pixels_by_color_wsi(slide, roi=None, lowerb=None, upperb=None, tile_size=2048, downsample=2, hole_size=0,
                              cores_num=4):
    """
    Segment regions based on their color profile in HSV space by sweeping over the whole slide image (WSI) tile by tile
    and return the total number of pixels in the segmented ROIs

    :param slide:        An openslide handle object to the whole slide image
    :param roi:          A list of cv2geojson.GeoContours for the selected ROIs
    :param lowerb:       inclusive lower bound array in HSV-space for color segmentation
    :param upperb:       inclusive upper bound array in HSV-space for color segmentation
    :param tile_size:    The tile size to sweep the whole slide image
    :param downsample:   Downsampling ratio as a power of two
    :param hole_size:    Remove holes smaller than hole_size; if zero, all holes will be reserved. if -1, all holes
                         will be filled in [micro-meter squared]
    :param cores_num:    Number of cores for parallel computation

    :return pixels_num:  total area of fat globules
    :return area:        total area of segmented ROI [milli-meter-squared]
    :return run_time:    Run time for the code in seconds
    """
    if lowerb is None:
        lowerb = [0, 0, 200]
    if upperb is None:
        upperb = [180, 25, 255]

    pixel_resolution = float(slide.properties['openslide.mpp-x'])
    # pixel_resolution = 10000.0 / float(slide.properties['tiff.XResolution'])

    # extract image tiles
    tiles = PatchGenerator(slide, tile_size=tile_size, overlap=0, downsample=downsample, roi=roi)
    print(f'Start parallel computation using {cores_num} cores...')
    start_time = time.time()
    with Pool(cores_num) as pool:
        partial_count_pixel_by_color = partial(count_pixels_by_color,
                                               lowerb=lowerb,
                                               upperb=upperb,
                                               hole_size=hole_size,
                                               resolution=pixel_resolution * downsample)
        results = pool.starmap(partial_count_pixel_by_color, tiles)
    pool.close()
    end_time = time.time()
    run_time = end_time - start_time
    print(f'parallel coding is completed in {np.round(run_time, 3)} seconds')

    # pooling results formatting as 'geojson' file
    pixels_num = np.sum(results)
    area = pixels_num * pixel_resolution * pixel_resolution * downsample * downsample * 1e-6
    return pixels_num, area, run_time


def fill_holes(mask, hole_size=-1.0, resolution=1.0):
    """
    Fill in small holes (< hole_size) inside the input 2D binary mask
    :param mask:         A binary numpy array [either 255 or 0]
    :param hole_size:    Remove holes smaller than hole_size; if zero, return mask. If -1, fill in all holes
    :param resolution:   The pixel resolution in micron

    :return mask:        A binary numpy array similar to input mask with holes are filled in.
    """
    if hole_size == 0:
        # return the input mask
        return mask

    # find contours
    contours, hierarchy = cv.findContours(mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    for index, contour in enumerate(contours):
        # check if the contour has a parent, which means it is a hole
        if hierarchy[0, index, 3] > -1:
            if hole_size < 0:
                # fill in the hole regardless of its size
                cv.drawContours(mask, [contour], 0, 255, -1)
            else:
                # compute the hole area
                area = cv.contourArea(contour) * resolution * resolution
                if area < hole_size:
                    # fill in the hole if its size is less than the given hole_size
                    cv.drawContours(mask, [contour], 0, 255, -1)
    return mask


def segment_foreground(img, lowerb=None, upperb=None, resolution=1.5, min_area=5e5):
    """
    segment the background in white color and return the foreground mask by negating the background

    :param img:          A numpy array of input image
    :param lowerb:       inclusive lower bound array in HSV-space for background segmentation
    :param upperb:       inclusive upper bound array in HSV-space for background segmentation
    :param resolution:   pixel resolution in micron
    :param min_area:     minimum area of a tissue ROI [micro-meter squared]; smaller ROIs will be removed

    :return foreground:  A binary mask represented as a numpy array of selected ROIs [either 255 or 0]
    """
    # extract background in white color
    if lowerb is None:
        lowerb = [0, 0, 230]
    if upperb is None:
        upperb = [180, 10, 255]

    # segment white regions as background
    background = segment_by_color(img, lowerb=lowerb, upperb=upperb)

    # pad background mask so holes touching the edges will be removed as well
    background = cv.copyMakeBorder(background, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=255)
    background = fill_holes(background, hole_size=min_area, resolution=resolution)
    foreground = cv.bitwise_not(background[1:-1, 1:-1])
    foreground = fill_holes(foreground, resolution=resolution)
    return foreground


@get_contours_decorator
def segment_foreground_contour(*args, **kwargs):
    # wrapper function for segment_foreground
    results = segment_foreground(*args, **kwargs)
    return results


def segment_foreground_wsi(slide, lowerb=None, upperb=None, min_area=5e5):
    """
    segment the background in white color and return the foreground mask contours (geojson)

    :param wsi:          An openslide object to the whole slide image
    :param lowerb:       inclusive lower bound array in HSV-space for background segmentation
    :param upperb:       inclusive upper bound array in HSV-space for background segmentation
    :param resolution:   pixel resolution in micron
    :param min_area:     minimum area of a tissue ROI [micro-meter squared]; smaller ROIs will be removed

    :return foreground:  A list of GeoContours
    """
    # extract background in white color
    if lowerb is None:
        lowerb = [0, 0, 230]
    if upperb is None:
        upperb = [180, 10, 255]

    # set downsample ratio
    downsample = 32

    # generate thumbnail image
    cols_num, rows_num = slide.dimensions
    cols_num = int(cols_num / downsample)
    rows_num = int(rows_num / downsample)
    img = get_tile_image(slide, (0, 0), (cols_num, rows_num), downsample)

    pixel_resolution = float(slide.properties['openslide.mpp-x']) * downsample
    geocontours = segment_foreground_contour(img,
                                             lowerb=lowerb,
                                             upperb=upperb,
                                             resolution=pixel_resolution,
                                             min_area=min_area)
    # scale contours
    for geocontour in geocontours:
        geocontour.scale_up(ratio=downsample)
    return geocontours


def separate_globules(mask, min_distance=5):
    """
    watershed segmentation to separate overlapping fat globules
    Params:
        mask: {numpy.ndarra}: binary mask either 255 or 0
        min_distance: {int}: specifies the minimum number of pixels separating the detected peaks in peak_local_max

    Returns:
        geocontours: {list: cv2geojson.GeoContour}: list of separated globules
    """
    distance = ndi.distance_transform_edt(mask == 255)
    coords = peak_local_max(distance, min_distance=min_distance)
    seeds = np.zeros(distance.shape, dtype=bool)
    seeds[tuple(coords.T)] = True
    markers, _ = ndi.label(seeds)
    markers = watershed(-distance, markers, mask=mask)

    mask_tmp = np.zeros_like(mask, dtype=np.uint8)
    geocontours = []
    for i in range(np.max(markers) + 1):
        mask_tmp[markers == i] = 255
        geocontours.extend(find_geocontours(mask_tmp, mode='imagej'))
        mask_tmp[markers == i] = 0
    return geocontours


def filter_fat_globules(geocontours, x_range=None, y_range=None, solidity=(0.7, 0.85), elongation=(0.05, 0.4),
                        diameter=(5.0, 100.0), resolution=1.0):
    """
    morphological filters to indentify fat globules as circular shapes
    Params:
        geocontours: {list: cv2geojson.GeoContours}

    Returns:
        globules: {list: cv2geojson.GeoContours}
        unknown: {list: cv2geojson.GeoContours}
        other: {list: cv2geojson.GeoContours}
    """
    if x_range is None:
        x_range = (0, np.max([np.max(cnt.contours[0]) for cnt in geocontours]))
    if y_range is None:
        y_range = (0, np.max([np.max(cnt.contours[0]) for cnt in geocontours]))

    globules = []
    unknown = []
    other = []
    for geocontour in geocontours:
        center, radius = geocontour.min_enclosing_circle()
        c_x, c_y = center
        d = radius * 2 * resolution
        e = geocontour.elongation()
        s = geocontour.solidity()

        # logical clauses
        is_fat_globule = e > elongation[1] and s > solidity[1] and (diameter[0] < d < diameter[1])
        is_unknown = e > elongation[0] and s > solidity[0] and d > diameter[0]
        if is_fat_globule:
            # assert this contour does not appear in the overlapped region
            if (x_range[0] <= c_x < x_range[1]) and (y_range[0] <= c_y < y_range[1]):
                globules.append(geocontour)
        elif is_unknown:
            # could be overlapping fat globules
            unknown.append(geocontour)
        else:
            other.append(geocontour)

    return globules, unknown, other
