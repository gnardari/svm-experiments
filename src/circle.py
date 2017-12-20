import numpy as np

def find_radius(dataset):
    center = np.mean(dataset, axis=0)
    max_dist = 0

    for p in dataset:
        d = np.linalg.norm(center-p)
        if d > max_dist:
            max_dist = d
    return max_dist
