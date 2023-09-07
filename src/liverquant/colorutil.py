"""colorutil
methods required to normalise staining in histology slides

[1] A method for normalizing histology slides for quantitative analysis,
    M Macenko, M Niethammer, JS Marron, D Borland, JT Woosley, G Xiaojun, C Schmitt, NE Thomas,
    IEEE ISBI, 2009. dx.doi.org/10.1109/ISBI.2009.5193250
"""
from functools import partial
from multiprocessing import Pool
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from sklearn.decomposition import NMF
import random
from .slidepatch import get_random_blocks


def get_ref_stian_vectors(stain='HE'):
    """
    Return the reference stain vectors
    Args:
        stain: {str} either 'HE', 'VG', 'PSR', or 'MTC'
    Returns:
        stain_vectors:
        max_concentrations:
    """
    if stain == 'HE':
        stain_vectors = np.array([[0.69652276, 0.32759918, 0.59774075],
                                  [0.70660026, 0.70872559, -0.66760321],
                                  [0.12478829, 0.62480942, 0.44386029]])
        max_concentrations = np.array([1.49931920, 1.36812984, 0.16090817])
    elif stain == 'PSR':
        stain_vectors = np.array([[0.29700838, 0.18553988, 0.95237073],
                                  [0.77096590, 0.23285926, -0.28187702],
                                  [0.56338051, 0.95464733, -0.11634148]])
        max_concentrations = np.array([1.82058375, 1.23927257, 0.31724941])
    elif stain == 'MTC':
        stain_vectors = np.array([[0.73060417, 0.22375230, 0.46580686],
                                  [0.64489721, 0.67064406, -0.71097224],
                                  [0.22433263, 0.70722801, 0.52682297]])
        max_concentrations = np.array([1.53293709, 1.51704764, 0.12945636])
    elif stain == 'VG':
        stain_vectors = np.array([[0.52941714, 0.12985407, 0.75251392],
                                  [0.71697463, 0.35222347, -0.64369966],
                                  [0.45350289, 0.92686382, 0.13918888]])
        max_concentrations = np.array([1.55067450, 1.20670689, 0.15414170])
    else:
        raise ValueError('Stain must be either "HE", "PSR", "MTC" or "VG".')
    return stain_vectors, max_concentrations


def get_fibrosis_hsv_bounds(stain='VG'):
    """
    Return the reference stain vectors
    Args:
        stain: {str} either 'HE', 'VG', 'PSR', or 'MTC'
    Returns:
        stain_vectors:
        max_concentrations:
    """
    if stain == 'VG':
        lowerb = [-15, 50, 100]
        upperb = [10, 255, 255]
    elif stain == 'PSR':
        lowerb = [-12, 50, 100]
        upperb = [12, 255, 255]
    elif stain == 'MTC':
        lowerb = [100, 50, 100]
        upperb = [140, 255, 255]
    else:
        raise ValueError('Stain must be either "PSR", "MTC" or "VG".')
    return lowerb, upperb


def compute_optical_density(img):
    """
    optical density: -log(I/I0)
    Args:
        img:
        max_intensity:

    Returns:
        optical_density
    """
    # Add 1 in case any pixels in the image have a value of 0 (log 0 is indeterminate)
    return -np.log((img.astype(np.float64) + 1) / 256)


def estimate_mixing_matrix(x, stain='HE', mode='SVD', alpha=1, beta=0.15):
    """

    Args:
        x: input rgb image as a numpy array of Nx3
        stain: the staining procedure: HE, PSR, MTC, or VG
        alpha: percentile of extreme angles to be included [default: 1]
        beta: remove transparent pixels with optical density (OD) below beta [default: 0.15]

    Returns:
        stain_vectors, i.e. the proportion of each wavelength absorbed from red, green, and blue channels
    """
    # estimate the optical density function: OD = I0 exp(VC)
    optical_density = compute_optical_density(x)

    # remove transparent pixels with OD intensity less than β (clear region with no tissue)
    non_transparent_pixels = np.all(optical_density > beta, axis=1)
    tissue_optical_density = optical_density[non_transparent_pixels,]

    if mode == 'SVD':
        # Calculate SVD on the OD tuples
        eigvals, eigvecs = np.linalg.eigh(np.cov(tissue_optical_density.T))

        # project data on the plane spanned by the eigenvectors corresponding to the two largest eigenvalues
        projections = tissue_optical_density.dot(eigvecs[:, 1:3])  # Dot product

        # Calculate angle of each point wrt the first eigenvector direction
        phi = np.arctan2(projections[:, 1], projections[:, 0])
        # find the min and max vectors and project back to OD space
        min_phi = np.percentile(phi, alpha)
        max_phi = np.percentile(phi, 100 - alpha)
        stain_vector_one = eigvecs[:, 1:3].dot(np.array([(np.cos(min_phi), np.sin(min_phi))]).T)
        stain_vector_two = eigvecs[:, 1:3].dot(np.array([(np.cos(max_phi), np.sin(max_phi))]).T)

        # construct the stain vector
        stain_vectors = np.array((stain_vector_one[:, 0], stain_vector_two[:, 0])).T

    elif mode == 'NMF':
        model = NMF(n_components=2, init='random', random_state=0)
        model.fit(tissue_optical_density)
        stain_vectors = model.components_
        stain_vectors = np.transpose(stain_vectors)
        stain_vectors = np.divide(stain_vectors, np.linalg.norm(stain_vectors, axis=0))
    else:
        raise ValueError('mode must be either "SVD" or "NMF".')

    # swap vectors if need be
    if stain == 'HE' and stain_vectors[0, 0] < stain_vectors[0, 1]:
        # a heuristic to make the vector corresponding to hematoxylin first and the one corresponding to eosin second
        stain_vectors = stain_vectors[:, [1, 0]]

    if stain == 'MTC' and stain_vectors[0, 0] < stain_vectors[0, 1]:
        stain_vectors = stain_vectors[:, [1, 0]]

    if stain == 'VG' and stain_vectors[2, 0] > stain_vectors[2, 1]:
        stain_vectors = stain_vectors[:, [1, 0]]

    if stain == 'PSR' and stain_vectors[1, 0] < stain_vectors[1, 1]:
        stain_vectors = stain_vectors[:, [1, 0]]

    # append the third stain vector as well
    stain_vector_three = np.cross(stain_vectors[:, 0], stain_vectors[:, 1])
    stain_vector_three /= np.linalg.norm(stain_vector_three)
    mixing_matrix = np.column_stack((stain_vectors, stain_vector_three))

    return mixing_matrix


def estimate_stain_concentration(img, mixing_matrix):
    """
    Estimate the stain saturation using pseudo-inverse
    Args:
        img:
        mixing_matrix:
        max_intensity:

    Returns:

    """
    img_vectorised = img.reshape(-1, 3)
    optical_density = compute_optical_density(img_vectorised)
    concentration = np.linalg.lstsq(mixing_matrix, np.transpose(optical_density), rcond=None)[0]
    concentration = np.transpose(concentration)
    concentration = concentration.reshape([img.shape[0], img.shape[1], mixing_matrix.shape[1]])
    return concentration


def separate_stains(img, mixing_matrix, scale=(1.0, 1.0, 1.0)):
    concentrations = estimate_stain_concentration(img, mixing_matrix)
    c1, c2, c3 = cv.split(concentrations)
    # normalise stains
    c1 /= scale[0]
    c2 /= scale[1]
    c3 /= scale[2]

    # recreate the image for each stains
    img1 = 255 * np.exp(-mixing_matrix[:, [0]].dot(c1.reshape(1, -1)))
    img1[img1 > 255] = 255
    img1 = np.reshape(img1.T, img.shape).astype(np.uint8)

    img2 = 255 * np.exp(-mixing_matrix[:, [1]].dot(c2.reshape(1, -1)))
    img2[img2 > 255] = 255
    img2 = np.reshape(img2.T, img.shape).astype(np.uint8)

    img3 = 255 * np.exp(-mixing_matrix[:, [2]].dot(c3.reshape(1, -1)))
    img3[img3 > 255] = 255
    img3 = np.reshape(img3.T, img.shape).astype(np.uint8)

    return img1, img2, img3


def normalise_stains(img, mixing_matrix, mixing_matrix_ref=None, scale=(1.0, 1.0, 1.0)):
    """

    Args:
        img:
        mixing_matrix:
        mixing_matrix_ref:
        scale:
        max_intensity:

    Returns:

    """
    if mixing_matrix_ref is None:
        mixing_matrix_ref = mixing_matrix
    concentrations = estimate_stain_concentration(img, mixing_matrix)

    concentrations_vectorised = concentrations.reshape(-1, mixing_matrix.shape[1])
    concentrations_normalised = np.multiply(concentrations_vectorised, scale)

    img_normal = 255 * np.exp(-mixing_matrix_ref.dot(concentrations_normalised.T))
    img_normal[img_normal > 255] = 255
    img_normal = np.reshape(img_normal.T, img.shape).astype(np.uint8)
    return img_normal


def get_maximum_stain_concentration(img, mask=None, mixing_matrix=None, q=99):
    """
    :param img_vectorised:
    :param stain_vectors:
    :param max_intensity:

    :return max_concentrations: numpy array of 2x1
    """
    concentrations = estimate_stain_concentration(img, mixing_matrix)

    c1mask = concentrations[:, :, 0] > concentrations[:, :, 1]
    c1mask = c1mask.astype(np.uint8) * 255
    if mask is not None:
        c1mask = cv.bitwise_and(c1mask, mask)
    c1max = np.percentile(concentrations[c1mask == 255, 0], q)

    c2mask = concentrations[:, :, 1] > concentrations[:, :, 0]
    c2mask = c2mask.astype(np.uint8) * 255
    if mask is not None:
        c2mask = cv.bitwise_and(c2mask, mask)
    c2max = np.percentile(concentrations[c2mask == 255, 1], q)

    if mask is not None:
        c3max = np.percentile(concentrations[mask == 255, 2], q)
    else:
        c3max = np.percentile(concentrations.reshape(-1, 1), q)
    c3max = np.max([0, c3max])
    max_concentrations = np.array([c1max, c2max, c3max])

    return max_concentrations


def estimate_mixing_matrix_wsi(slide, stain=None, mode='SVD', alpha=1, beta=0.15, roi=None, downsample=1, blocks_num=50):
    img, mask = get_random_blocks(slide, blocks_num=blocks_num, roi=roi, downsample=downsample)
    x = img[mask == 255, ]
    mixing_matrix = estimate_mixing_matrix(x, stain=stain, mode=mode, alpha=alpha, beta=beta)
    return mixing_matrix


def get_maximum_stain_concentration_wsi(slide, mixing_matrix, roi=None, downsample=1, blocks_num=50, q=99):
    """
    :param img_vectorised:
    :param stain_vectors:
    :param max_intensity:

    :return max_concentrations: numpy array of 2x1
    """
    img, mask = get_random_blocks(slide, blocks_num=blocks_num, roi=roi, downsample=downsample)
    max_concentrations = get_maximum_stain_concentration(img, mask=mask, mixing_matrix=mixing_matrix, q=q)
    return max_concentrations


if __name__ == '__main__':
    img_a = cv.imread('C:\\Users\\mfarzi\\Documents\\mycodes\\liverquant\\example\\normalisation\\MTC_tile_01.jpg')
    img_a = cv.cvtColor(img_a, cv.COLOR_BGR2RGB)

    img_b = cv.imread('C:\\Users\\mfarzi\\Documents\\mycodes\\liverquant\\example\\normalisation\\HE_tile_01.jpg')
    img_b = cv.cvtColor(img_b, cv.COLOR_BGR2RGB)

    img_vectorised_a = img_a.reshape(-1, 3)
    img_vectorised_b = img_b.reshape(-1, 3)
    # M = np.array([[0.65,  0.70, 0.29],
    #               [0.07,  0.99, 0.11],
    #               [0.27,  0.57, 0.78]])
    Ma = estimate_mixing_matrix(img_vectorised_a, stain='MTC', mode='NMF')
    scale_a = get_maximum_stain_concentration(img_a, Ma)
    Mb = estimate_mixing_matrix(img_vectorised_b, stain='PSR', mode='NMF')
    scale_b = get_maximum_stain_concentration(img_b, Mb)
    img1, img2, img3 = separate_stains(img_a, Ma)
    plt.subplot(1, 4, 1)
    plt.imshow(img_a)
    plt.subplot(1, 4, 2)
    plt.imshow(img1)
    plt.subplot(1, 4, 3)
    plt.imshow(img2)
    plt.subplot(1, 4, 4)
    plt.imshow(img3)
    #
    m1, m2, m3 = separate_stains(img_b, Mb)
    plt.figure()
    plt.subplot(1, 4, 1)
    plt.imshow(img_b)
    plt.subplot(1, 4, 2)
    plt.imshow(m1)
    plt.subplot(1, 4, 3)
    plt.imshow(m2)
    plt.subplot(1, 4, 4)
    plt.imshow(m3)

    scale = scale_a / scale_b
    # scale[2] = 0
    img_c = normalise_stains(img_b, Mb, Ma, scale)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img_b)

    plt.subplot(1, 2, 2)
    plt.imshow(img_c)
