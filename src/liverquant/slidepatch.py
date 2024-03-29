import numpy as np
from PIL import Image
from cv2geojson import draw_geocontours
import random


class PatchGenerator:
    """
    PatchGenerator returns an iterator to sweep through a whole slide image tile by tile
    """

    def __init__(self, slide, tile_size=1024, overlap=0, downsample=1, roi=None, fov=None, ensure_fit=False):
        if fov is None:
            fov = [(0, 0), slide.dimensions]
        tile_addresses = extract_tiles(fov, tile_size=tile_size, overlap=overlap, downsample=downsample,
                                       roi=roi, ensure_fit=ensure_fit)

        self.slide = slide
        self.address = [address[0] for address in tile_addresses]
        self.local_address = [address[1] for address in tile_addresses]
        self.start = 0
        self.end = len(tile_addresses) - 1
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


def extract_tiles(frame_size, tile_size=(1024, 1024), overlap=(0, 0), downsample=1, roi=None, ensure_fit=False):
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

    # generate low-res mask
    # roi_mask = get_tile_mask(roi=roi, address=(col_offset, row_offset), tile_size=(col_max, row_max), downsample=32)

    if ensure_fit:
        col_max -= stride_x_downsample
        row_max -= stride_y_downsample
    address = []
    for col in range(col_offset, col_max, stride_x_downsample):
        for row in range(row_offset, row_max, stride_y_downsample):
            if roi is None:
                address.append([(col, row), (int((col - col_offset) / stride_x_downsample),
                                             int((row - row_offset) / stride_y_downsample))])
            else:
                # check if this tile overlaps with the ROI
                # mask = roi_mask[int(row / 32):int((row + tile_size[1]) / 32),
                #        int(col / 32):int((col + tile_size[0]) / 32)]
                mask = get_tile_mask(roi, (col, row), tile_size, downsample)
                if 255 in mask[overlap[1]:tile_size[1]-overlap[1], overlap[0]:tile_size[0]-overlap[0]]:
                # if 255 in mask:
                    address.append([(col, row), (int((col - col_offset) / stride_x_downsample),
                                                 int((row - row_offset) / stride_y_downsample))])
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
    level_downsamples = np.rint(slide.level_downsamples)
    level = np.where(level_downsamples <= downsample)[0][-1]
    # level = slide.get_best_level_for_downsample(downsample)

    # find the downsample ratio at the best level
    downsample_ratio = downsample / level_downsamples[level]

    # find the tile_size at the best level
    width = int(tile_size[0] * downsample_ratio)
    height = int(tile_size[1] * downsample_ratio)

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
        # roi_scaled = [contour.scale_down(ratio=downsample, offset=address) for contour in roi]
        draw_geocontours(mask, roi, scale=downsample, offset=address, mode='imagej')
    return mask


def get_random_blocks(slide, blocks_num=50, roi=None, downsample=1, block_size=1024):
    tiles = extract_tiles(((0, 0), slide.dimensions),
                          tile_size=(block_size, block_size),
                          overlap=(0, 0),
                          downsample=downsample,
                          roi=roi,
                          ensure_fit=False)

    samples_num = np.min([blocks_num, len(tiles)])
    tiles_random = random.sample(tiles, samples_num)

    img = np.zeros((block_size, 0, 3), dtype=np.uint8)
    mask = np.zeros((block_size, 0), dtype=np.uint8)
    for address in tiles_random:
        tile = get_tile_image(slide,
                              address=address[0],
                              tile_size=(block_size, block_size),
                              downsample=downsample)
        img = np.concatenate((img, tile), axis=1)
        tile_mask = get_tile_mask(roi,
                                  address=address[0],
                                  tile_size=block_size,
                                  downsample=downsample)
        mask = np.concatenate((mask, tile_mask), axis=1)

    return img, mask


def get_thumbnail(slide, downsample=32):
    # generate thumbnail image
    cols_num, rows_num = slide.dimensions
    cols_num = int(cols_num / downsample)
    rows_num = int(rows_num / downsample)
    img = get_tile_image(slide, (0, 0), (cols_num, rows_num), downsample)
    return img
