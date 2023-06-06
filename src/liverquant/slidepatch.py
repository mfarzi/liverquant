import os
import numpy as np
from PIL import Image
import cv2 as cv
openslide_path = "C:\\Users\\mfarzi\\Documents\\mycodes\\Libraries\\openslide-win64-20220811\\bin"
os.environ['PATH'] = openslide_path + ";" + os.environ['PATH']
from openslide import OpenSlide
from cv2geojson import draw_geocontours


class PatchGenerator:
    """
    PatchGenerator returns an iterator to sweep through a whole slide image tile by tile
    """
    def __init__(self, slide, tile_size=1024, overlap=0, downsample=1, roi=None, fov=None):
        if fov is None:
            fov = [(0, 0), slide.dimensions]
        tile_addresses = extract_tiles(fov, tile_size=tile_size, overlap=overlap, downsample=downsample,
                                       roi=roi)

        self.slide = slide
        self.address = [address[0] for address in tile_addresses]
        self.local_address = [address[1] for address in tile_addresses]
        self.start = 0
        self.end = len(tile_addresses)-1
        self.current = 0
        self.tile_size = tile_size
        self.downsample = downsample
        self.roi = roi

    def __iter__(self):
        return self

    def __next__(self):
        if self.current > self.end:
            raise StopIteration
        address = self.address[self.current]
        patch = get_tile_image(self.slide, address, self.tile_size, self.downsample)
        self.current += 1
        if self.roi is None:
            mask = None
        else:
            mask = get_tile_mask(self.roi, address, self.tile_size, self.downsample)
        return patch, mask


def extract_tiles(frame_size, tile_size=(1024, 1024), overlap=(0, 0), downsample=1, roi=None):
    """
    Export image tiles' coordinates from the whole slide image (WSI) for patch-based analysis

    :param frame_size: the native dimensions of the whole slide image (WSI) at level 0 in the image pyramid; (col , row)
    :param tile_size:  the size of each tile used to sweep the WSI
    :param overlap:    the amount of overlap between adjacent tiles
    :param downsample: the downsampling factor as power of two
    :param roi:        a list of contours (numpy arrays) identifying the regions of interest (ROIs) within the level 0

    :return address: a list of tuples (col, row) representing the top-left corner of each tile
    """
    # check if tile size and the overlap are scalar
    if np.isscalar(tile_size):
        tile_size = (tile_size, tile_size)

    if np.isscalar(overlap):
        overlap = (overlap, overlap)

    col_offset, row_offset = frame_size[0]
    # extract patches
    col_max, row_max = frame_size[1]
    stride_x = tile_size[0] - 2 * overlap[0]
    stride_y = tile_size[1] - 2 * overlap[1]
    stride_x_downsample = stride_x * downsample
    stride_y_downsample = stride_y * downsample

    address = []
    for col in range(col_offset, col_max, stride_x_downsample):
        for row in range(row_offset, row_max, stride_y_downsample):
            if roi is None:
                address.append([(col, row), (int((col-col_offset)/stride_x_downsample), int((row-row_offset)/stride_y_downsample))])
            else:
                # check if this tile overlaps with the ROI
                mask = get_tile_mask(roi, (col, row), tile_size, downsample)
                if 255 in mask[overlap[1]:tile_size[1]-overlap[1], overlap[0]:tile_size[0]-overlap[0]]:
                    address.append([(col, row), (int((col-col_offset)/stride_x_downsample), int((row-row_offset)/stride_y_downsample))])
    return address


def get_tile_image(slide, address, tile_size, downsample=1):
    """
    Helper function for openslide-python library to return an RGB image for a tile (numpy array)

    :param slide:      a slide object
    :param address:    coordinates of the top-left corner of the tile within the level 0 (native resolution) as a
                       (col, row) tuple
    :param tile_size:  the width and height of the tile as a (width, height) tuple within the downsampled resolution
    :param downsample: downsample factor as a power of two

    :return img: RGB image of the tile as a numpy array
    """
    if np.isscalar(tile_size):
        tile_size = (tile_size, tile_size)

    # find the best level to read the image
    level = slide.get_best_level_for_downsample(downsample)

    # find the downsample ratio at the best level
    downsample_ratio = int(downsample/np.rint(slide.level_downsamples[level]))

    # find the tile_size at the best level
    width = tile_size[0] * downsample_ratio
    height = tile_size[1] * downsample_ratio

    # read the image tile
    tile = slide.read_region(address, level, (width, height))
    if downsample_ratio > 1:
        tile.thumbnail(tile_size, Image.Resampling.LANCZOS)
    tile_rgb = np.array(tile.convert('RGB'))
    return tile_rgb


def get_tile_mask(roi=None, address=(0, 0), tile_size=(1024, 1024), downsample=1):
    """
    Return a binary mask (numpy array) where pixels inside the ROI are 255; otherwise, 0

    :param roi:        a list of contours (numpy arrays) identifying the regions of interest (ROIs) within the level 0
                       (native resolution)
    :param address:    coordinates of the top-left corner of the tile within the level 0 (native resolution) as a
                       (col, row) tuple
    :param tile_size:  the width and height of the tile as a (height, width) tuple within the downsampled resolution
    :param downsample: downsample factor as a power of two

    :return mask: Binary mask (numpy array) where pixels inside the ROI are 255; otherwise, 0
    """
    # mask = np.zeros(tile_size, dtype=np.uint8)
    if np.isscalar(tile_size):
        tile_size = (tile_size, tile_size)
    width = tile_size[0]
    height = tile_size[1]
    mask = np.zeros((height, width), dtype=np.uint8)
    if roi is None:
        mask = mask + 255
    else:
        roi_scaled = [contour.scale_down(ratio=downsample, offset=address) for contour in roi]
        mask = draw_geocontours(mask, roi_scaled, mode='imagej')
    return mask
