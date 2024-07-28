import numpy as np
import itertools, h5py, pickle
import os
from spectra_topology_utils import PosLoG, Phi_image, Phi_graph
import cv2
from skimage.filters import (threshold_triangle, threshold_li, threshold_yen,
                             threshold_minimum, threshold_isodata, threshold_mean,
                             sobel_v, sobel_h)
from scipy.spatial.distance import pdist, squareform

def apply_kernel(image, kernel):
    return cv2.filter2D(image, -1, kernel)

def apply_threshold(image, method):
    threshold_methods = {
        'adaptive_mean': lambda img: cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2),
        'adaptive_gaussian': lambda img: cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
        'mean': lambda img: img >= threshold_mean(img),
        'triangle': lambda img: img >= threshold_triangle(img),
        'li': lambda img: img >= threshold_li(img),
        'yen': lambda img: img >= threshold_yen(img),
        'minimum': lambda img: img >= threshold_minimum(img),
        'isodata': lambda img: img >= threshold_isodata(img)
    }

    if method not in threshold_methods:
        raise ValueError("Unknown method")
    
    return threshold_methods[method](image)

def get_max(image, method, alpha=1):
    i_out = []
    j_out = []
    image_temp = image.copy()

    # Apply the chosen thresholding method
    threshold_image = apply_threshold(image_temp, method)
    if isinstance(threshold_image, np.ndarray):
        threshold = alpha * np.std(image)
    else:
        threshold = alpha * threshold_image

    # Find all indices where image_temp is greater than or equal to the threshold
    indices = np.argwhere(image_temp >= threshold)

    if indices.size == 0:
        return i_out, j_out

    # Sort indices by image value in descending order
    indices = indices[np.argsort(-image_temp[indices[:, 0], indices[:, 1]])]

    for idx in indices:
        j, i = idx
        if image_temp[j, i] >= threshold:
            i_out.append(i)
            j_out.append(j)
            # Suppress non-maximum peaks in a 3x3 neighborhood
            image_temp[max(0, j-2):min(image_temp.shape[0], j+3), max(0, i-2):min(image_temp.shape[1], i+3)] = 0

    # Remove duplicates based on proximity
    if len(i_out) > 0:
        coordinates = np.array(list(zip(i_out, j_out)))
        dist_matrix = squareform(pdist(coordinates))
        np.fill_diagonal(dist_matrix, np.inf)
        to_remove = set()

        for i in range(len(coordinates)):
            if i not in to_remove:
                for j in range(i + 1, len(coordinates)):
                    if dist_matrix[i, j] < 4:  # Threshold for proximity
                        to_remove.add(j)

        i_out = [i for idx, i in enumerate(i_out) if idx not in to_remove]
        j_out = [j for idx, j in enumerate(j_out) if idx not in to_remove]
 
    return i_out, j_out

def auto_Emax(c, emax=20, elen=64, e_padding=None):
    dos = PosLoG(Phi_image(c, Emax=emax, Elen=elen))
    
    for_x = sobel_v(dos)
    for_y = sobel_h(dos)
    filtered = np.sqrt(np.square(for_x) + np.square(for_y))
    
    # Get the peaks
    i, j = get_max(filtered, method="triangle")
    
    # Determine new bounds
    E_range = np.linspace(-emax, emax, elen)
    min_re, max_re = abs(E_range[np.min(i)]), abs(E_range[np.max(i)])
    min_im, max_im = abs(E_range[np.min(j)]), abs(E_range[np.max(j)])
    
    boundary = np.max([min_re, max_re, min_im, max_im])
    if e_padding is None: boundary *= 1.005
    else: boundary += e_padding
    
    return boundary


def generate_dataset_graph(file_name, samples_per_dim=7, dim=4, c_max=1.2, Elen=256):
    # Generate all combinations of coefficients
    values = list(np.linspace(-c_max, c_max, samples_per_dim))
    combinations = list(itertools.product(values, repeat=dim))
    
    # Prepare data structures
    graphs = []
    labels = []
    
    for comb in combinations:
        if dim == 4:
            c = [1, comb[0], comb[1], 0, comb[2], comb[3], 1]
        if dim == 6:
            c = [1, comb[0], comb[1], comb[2], 0, comb[3], comb[4], comb[5], 1]
        ### With Boxing method ###
        Eauto = auto_Emax(c, emax=20, elen=64)
        graph = Phi_graph(c, Emax=Eauto, Elen=Elen)
        ### With Boxing method ###
        graphs.append(graph)
        labels.append(comb)
    
    labels = np.array(labels)
    
    # Serialize graphs using pickle
    serialized_graphs = [pickle.dumps(graph) for graph in graphs]
    
    # Save the dataset
    with h5py.File(file_name, 'w') as f:
        f.create_dataset('graphs', data=np.string_(serialized_graphs))
        f.create_dataset('labels', data=labels)

def load_dataset_graph(file_name):
    with h5py.File(file_name, 'r') as f:
        serialized_graphs = f['graphs'][:]
        labels = f['labels'][:]

    graphs = [pickle.loads(graph.tobytes()) for graph in serialized_graphs]
    return graphs, labels

if __name__ == '__main__':
    if not os.path.exists('./Datasets'):
        os.makedirs('./Datasets')
    generate_dataset_graph('./Datasets/dataset_graph_dim6.h5', samples_per_dim=7, dim=6, c_max=1.2, Elen=1024)