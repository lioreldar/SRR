import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as spla
import shapely.geometry as shapes
from scipy.stats import ortho_group

import GMM_Params


# sample a random PSD matrix with prescribed ratio of max_ev / min_ev
def random_PSD():
    diag = np.zeros((2, 2))
    diag[0, 0] = 3
    diag[1, 1] = diag[0, 0] * GMM_Params.ellpsoid_ratio
    random_unitary = ortho_group.rvs(2)
    covMat = random_unitary @ diag @ random_unitary.T
    return covMat


# given an arbitrary vector in R^2 compute its reflection that
# is contained in the sector between [-pi/2, pi/2]
def shift_to_positive_orthant(vector):
    if np.sign(vector[0]) != np.sign(vector[1]):
        vector[0] = np.abs(vector[0])
        vector[1] = -1 * np.abs(vector[1])
    else:
        vector[0] = np.abs(vector[0])
        vector[1] = np.abs(vector[1])

    return vector


# sampled from a Gaussian distribution with prescribed covariance, mean
def sample_Gaussian(covariance_matrix, mean):
    noisy_vector = None
    lower_triangular = spla.cholesky(covariance_matrix)

    for idx in range(1000):
        zero_mean_iid_gaussian = np.random.normal(size=(1, 2))
        zero_mean_iid_gaussian = zero_mean_iid_gaussian[0]
        new_sample = mean + zero_mean_iid_gaussian.dot(lower_triangular)
        dummy = np.zeros((1, 2))
        dummy[0] = new_sample
        if noisy_vector is None:
            noisy_vector = dummy
        else:
            noisy_vector = np.append(noisy_vector, dummy, axis=0)

    return noisy_vector


# generate a dataset from a Gaussian mixture with random covariance / mean
def generate_dataset(number_of_clusters, mean_to_variance_ratio):
    plt.figure()
    dataset = None

    for idx in range(number_of_clusters):

        # sample random covariance / mean
        covariance_matrix = random_PSD()
        mean = mean_to_variance_ratio * np.random.uniform(0, 10, 2)

        # sample dataset with these parameters
        blob_dist = sample_Gaussian(covariance_matrix, mean)
        plt.scatter(blob_dist[:, 0], blob_dist[:, 1], s=5)
        if dataset is None:
            dataset = blob_dist
        else:
            dataset = np.concatenate((dataset, blob_dist))

    return dataset


def create_polygon_samples(polygon):
    samples = []

    # generate samples inside the polygon with rejection sampling
    while len(samples) < GMM_Params.polygon_num_samples:
        coords = np.random.uniform(0, 50, (2))
        point = shapes.Point(coords)
        if polygon.contains(point):
            samples.append(coords)

    samples = np.array(samples)
    plt.figure()
    plt.scatter(samples[:, 0], samples[:, 1], s=5)

    return samples
