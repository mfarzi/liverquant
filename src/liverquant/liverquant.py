import numpy as np
import cv2 as cv
from cv2geojson import find_geocontours, draw_geocontours, export_annotations
from functools import wraps, partial
from .slidepatch import PatchGenerator, get_tile_image, get_random_blocks
from .colorutil import get_ref_stian_vectors, estimate_mixing_matrix, get_maximum_stain_concentration, normalise_stains,\
    estimate_mixing_matrix_wsi, get_maximum_stain_concentration_wsi, get_fibrosis_hsv_bounds
from multiprocessing import Pool
import time
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from sklearn.mixture import GaussianMixture


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
    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    # mask_white = cv.morphologyEx(mask_white, cv.MORPH_OPEN, kernel=kernel, iterations=3)

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


def segment_foreground(img, lowerb=None, upperb=None, resolution=1.0, min_area=5e5, mode='bg'):
    """
    segment the background in white color and return the foreground mask by negating the background or segment the
    foreground directly if stain is provided

    :param img:          A numpy array of input image
    :param lowerb:       inclusive lower bound array in HSV-space for background segmentation
    :param upperb:       inclusive upper bound array in HSV-space for background segmentation
    :param resolution:   pixel resolution in micron
    :param min_area:     minimum area of a tissue ROI [micro-meter squared]; smaller ROIs will be removed
    :param mode:        'bg' for background segmentation or 'fg' for foreground segmentation

    :return foreground:  A binary mask represented as a numpy array of selected ROIs [either 255 or 0]
    """
    # extract background in white color
    if lowerb is None:
        if mode == 'bg':
            lowerb = [0, 0, 230]
        elif mode == 'fg':
            lowerb = [140, 25, 180]
        else:
            raise ValueError('mode must be either "bg" or "fg".')

    if upperb is None:
        if mode == 'bg':
            upperb = [180, 10, 255]
        elif mode == 'fg':
            upperb = [180, 255, 255]
        else:
            raise ValueError('mode must be either "bg" or "fg".')

    if mode == 'bg':
        # segment white regions as background
        background = segment_by_color(img, lowerb=lowerb, upperb=upperb)

        # pad background mask so holes touching the edges will be removed as well
        background = cv.copyMakeBorder(background, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=255)
        background = fill_holes(background, hole_size=min_area, resolution=resolution)
        foreground = cv.bitwise_not(background[1:-1, 1:-1])
        foreground = fill_holes(foreground, resolution=resolution)
    elif mode == 'fg':
        # segment foreground using color segmentation
        mask = segment_by_color(img, lowerb=lowerb, upperb=upperb, hole_size=-1, resolution=resolution)
        # filter small blobs
        foreground = np.zeros_like(mask, dtype=np.uint8)
        geocontours = find_geocontours(mask)
        for geocontour in geocontours:
            if geocontour.area(resolution=resolution) > min_area:
                draw_geocontours(foreground, [geocontour])
    else:
        raise ValueError('mode must be either "bg" or "fg".')

    return foreground


@get_contours_decorator
def segment_foreground_contour(*args, **kwargs):
    # wrapper function for segment_foreground
    results = segment_foreground(*args, **kwargs)
    return results


def segment_foreground_wsi(slide, lowerb=None, upperb=None, min_area=5e5, mode='bg'):
    """
    segment the background in white color and return the foreground mask contours (geojson)

    :param wsi:          An openslide object to the whole slide image
    :param lowerb:       inclusive lower bound array in HSV-space for background segmentation
    :param upperb:       inclusive upper bound array in HSV-space for background segmentation
    :param resolution:   pixel resolution in micron
    :param min_area:     minimum area of a tissue ROI [micro-meter squared]; smaller ROIs will be removed
    :param mode:        'bg' for background segmentation or 'fg' for foreground segmentation
    :return foreground:  A list of GeoContours
    """
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
                                             min_area=min_area,
                                             mode=mode)
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


def segment_fibrosis_contour(img, mask=None, stain=None, lowerb=None, upperb=None, mixing_matrix=None,
                             ref_mixing_matrix=None, scale=None, hole_size=0.0, blob_size=0.0, resolution=1.0):
    """
    Segment regions based on their color profile in HSV space

    :param img:          A numpy array of input image tile in RGB space
    :param mask:         A numpy array of desired ROIs [either 255 or 0] corresponding to img
    :param stain:        VG or PSR or MTC
    :param lowerb:       Inclusive lower bound array in HSV-space for color segmentation
    :param upperb:       Inclusive upper bound array in HSV-space for color segmentation
    :param mixing_matrix:
    :param ref_mixing_matrix:
    :param scale:
    :param hole_size:    Remove holes smaller than hole_size; if zero, all holes will be reserved. If -1, all holes
                         will be removed.
    :param blob_size:
    :param resolution:   Image pixel resolution in micron

    :return geocontours: A list of cv2geojson.GeoContour objects representing color segmented ROIs
    """
    if mask is None:
        mask = np.zeros(img.shape[:2], dtype=np.uint8)+255

    if scale is None:
        scale = (1, 1, 0)

    # retrieve lower and upper bounds
    if lowerb is None:
        lowerb = get_fibrosis_hsv_bounds(stain)[0]
    if upperb is None:
        upperb = get_fibrosis_hsv_bounds(stain)[1]

    # retrieve reference measurements
    if ref_mixing_matrix is None:
        # no image normalisation
        img_normal = img
    else:
        if mixing_matrix is None:
            x = img[mask == 255,]
            mixing_matrix = estimate_mixing_matrix(x, stain=stain, mode='SVD', alpha=1, beta=0.15)
        # normalise the image
        img_normal = normalise_stains(img, mixing_matrix=mixing_matrix, mixing_matrix_ref=ref_mixing_matrix, scale=scale)

    contours = segment_by_color_contour(img_normal,
                                        mask=mask,
                                        lowerb=lowerb,
                                        upperb=upperb,
                                        hole_size=hole_size,
                                        resolution=resolution)
    # filter small blob size
    geocontours = [cnt for cnt in contours if cnt.area(resolution) > blob_size]

    return geocontours


def segment_fibrosis_wsi(slide, stain, roi=None, lowerb=None, upperb=None, mixing_matrix=None, ref_mixing_matrix=None,
                         scale=None, hole_size=0.0, blob_size=0.0, tile_size=2048, downsample=8, cores_num=4):
    """
    segment fibrosis in liver tissue using color and morphological features by sweeping over the whole slide image
    (WSI) tile by tile and return annotations (geojson)

    :param slide:                An openslide object to the whole slide image
    :param roi:                  A list of cv2geojson.GeoContours for the selected ROIs
    :param stain
    :param lowerb:               Inclusive lower bound array in HSV-space for color segmentation
    :param upperb:               Inclusive upper bound array in HSV-space for color segmentation
    :param mixing_matrix:
    :param ref_mixing_matrix:
    :param tile_size:            The tile size to sweep the whole slide image
    :param downsample:           Downsampling ratio as a power of two
    :param cores_num:            Number of cores for parallel computation

    :return cpa:                 The amount of collagen in the tissue represented in percent
    :return geocontours:         List of cv2geojson.GeoContour geometries representing fat globules as Polygons
    :return roi:                 list of cv2geojson.GeoContour geometries representing the foreground mask
    """
    if roi is None:
        roi = segment_foreground_wsi(slide)

    # extract image tiles
    tiles = PatchGenerator(slide, tile_size=tile_size, overlap=0, downsample=downsample, roi=roi)

    pixel_resolution = float(slide.properties['openslide.mpp-x'])

    print(f'Start parallel computation using {cores_num} cores...')
    with Pool(cores_num) as pool:
        partial_segment_fibrosis = partial(segment_fibrosis_contour,
                                           stain=stain,
                                           lowerb=lowerb,
                                           upperb=upperb,
                                           mixing_matrix=mixing_matrix,
                                           ref_mixing_matrix=ref_mixing_matrix,
                                           scale=scale,
                                           hole_size=hole_size,
                                           blob_size=blob_size,
                                           resolution=pixel_resolution*downsample)
        results = pool.starmap(partial_segment_fibrosis, tiles)
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

    # estimate collagen proportionate area (cpa)
    area_tissue = np.sum([cnt.area(pixel_resolution) for cnt in roi])
    area_fibrosis = np.sum([cnt.area(pixel_resolution) for cnt in geocontours])
    cpa = np.round(area_fibrosis / area_tissue * 100, 2)

    return cpa, geocontours, roi


def segment_fibrosis(*args, **kwargs):
    # retrieve contours for fibrotic tissue
    geocontours = segment_fibrosis_contour(*args, **kwargs)

    foreground = kwargs.get('mask')
    if foreground is None:
        frame_size = args[0].shape[:2]
        foreground = np.zeros(frame_size, dtype=np.uint8)+255

    # draw mask
    collagen_mask = np.zeros(foreground.shape, dtype=np.uint8)
    draw_geocontours(collagen_mask, geocontours, mode='imagej')

    cpa = np.round(np.count_nonzero(collagen_mask) / np.count_nonzero(foreground) * 100, 2)
    return cpa, collagen_mask, foreground


def get_hsv_bounds(img, mask=None, stain='VG', outlier_rate=2.5, beta=10, means_init=None, n_iter=1, verbose=False):
    # extract white regions and exclude them
    mask_white = segment_by_color(img, mask, lowerb=[0, 0, 200], upperb=[180, beta, 255])
    mask_tissue = cv.bitwise_not(mask_white)
    if mask is not None:
        mask_tissue = cv.bitwise_and(mask_tissue, mask)

    # convert to HSV
    img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    h, s, v = cv.split(img_hsv)
    hue = np.int32(h)
    if stain in ['VG', 'PSR']:
        hue[hue > 90] = hue[hue > 90] - 180
    x = hue[mask_tissue == 255]

    # remove outliers
    q1 = np.quantile(x, 0.25)
    q3 = np.quantile(x, 0.75)
    iqr = q3 - q1

    # estimate lower bound
    lower_bound = np.percentile(x, outlier_rate)
    lower_bound_range = np.array([q1 - 1.5 * iqr, q1 - 3 * iqr, q1 - 4.5 * iqr, q1 - 6 * iqr, q1 - 7.5 * iqr, q1 - 9 * iqr])
    if np.any(lower_bound_range < lower_bound):
        lower_bound = np.max(lower_bound_range[lower_bound_range < lower_bound])

    # estimate upper bound
    upper_bound = np.percentile(x, 100-outlier_rate)
    upper_bound_range = np.array([q3 + 1.5 * iqr, q3 + 3 * iqr, q3 + 4.5 * iqr, q1 + 6 * iqr, q1 + 7.5 * iqr, q1 + 9 * iqr])
    if np.any(upper_bound_range > upper_bound):
        upper_bound = np.min(upper_bound_range[upper_bound_range > upper_bound])

    bounds = [lower_bound, upper_bound]

    # remove outliers
    y = np.array([xi for xi in x if bounds[0] < xi < bounds[1]])
    y = np.reshape(y, (-1, 1))

    best_model = GaussianMixture(n_components=2, means_init=means_init)
    best_model.fit(y)
    loglikelihood = best_model.score(y)
    for i in range(n_iter):
        gmm = GaussianMixture(n_components=2, means_init=means_init)
        gmm.fit(y)
        score = gmm.score(y)
        if score < loglikelihood:
            best_model = gmm
            loglikelihood = score
        if verbose:
            print(f'iteration {i}: loglikelihood = {score}')

    # find threshold for prob = 0.5
    xx = np.linspace(bounds[0], bounds[1], 1000)
    probs = best_model.predict_proba(xx.reshape(-1, 1))
    thresh = np.min(xx[np.where(np.abs(probs[:, 0] - 0.5) < 0.05)])

    mu1, mu2 = best_model.means_.flatten()
    sigma1, sigma2 = np.sqrt(best_model.covariances_).flatten()

    if mu1 < mu2:
        lowerb = [int(mu1 - 3 * sigma1), 50, 100]
    else:
        lowerb = [int(mu2 - 3 * sigma2), 50, 100]
    upperb = [int(thresh), 255, 255]

    if verbose:
        print(f'best model: mu1 = {mu1}, mu2 = {mu2}, sigma1 = {sigma1}, sigma2 = {sigma2}, weights = {best_model.weights_}, bounds = {bounds}')
    return lowerb, upperb


def get_hsv_bounds_wsi(slide, roi=None, stain='VG', blocks_num=50, downsample=32, outlier_rate=2.5, beta=10,
                       means_init=None, n_iter=1, verbose=False):
    img, mask = get_random_blocks(slide, blocks_num=blocks_num, roi=roi, downsample=downsample)
    lowerb, upperb = get_hsv_bounds(img,
                                    mask=mask,
                                    stain=stain,
                                    outlier_rate=outlier_rate,
                                    beta=beta,
                                    means_init=means_init,
                                    n_iter=n_iter,
                                    verbose=verbose)
    return lowerb, upperb

