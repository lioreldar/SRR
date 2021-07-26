import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as spla
import shapely.geometry as shapes
from matplotlib.patches import Rectangle
from scipy.stats import ortho_group
from sklearn import mixture

import GMM_Params
from BoundingRect import BoundingRect


def shift_to_positive_orthant(vector):
    if np.sign(vector[0]) != np.sign(vector[1]):
        vector[0] = np.abs(vector[0])
        vector[1] = -1 * np.abs(vector[1])
    else:
        vector[0] = np.abs(vector[0])
        vector[1] = np.abs(vector[1])

    return vector

def me_be_test_func():
    pass

def generate_dist(covMat, mean):
    noisy_vector = None
    lower_triangular = spla.cholesky(covMat)

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


def create_polygon_samples():
    polygon = shapes.Polygon([(0, 0), (50, 0), (50, 50), (0, 0)])
    samples = []
    # generate samples inside the polygon with rejection sampling
    while len(samples) < 10000:
        coords = np.random.uniform(0, 50, (2))
        point = shapes.Point(coords)
        if polygon.contains(point):
            samples.append(coords)

    samples = np.array(samples)
    plt.figure()
    plt.scatter(samples[:, 0], samples[:, 1], s=5)

    return samples


def generate_dataset(number_of_clusters, mean_to_variance_ratio):
    plt.figure()
    dataset = None

    for idx in range(number_of_clusters):

        # sample random covariance / mean
        covMat = random_PSD()
        mean = mean_to_variance_ratio * np.random.uniform(0, 10, 2)

        # sample dataset with these parameters
        blob_dist = generate_dist(covMat, mean)
        plt.scatter(blob_dist[:, 0], blob_dist[:, 1], s=5)
        if dataset is None:
            dataset = blob_dist
        else:
            dataset = np.concatenate((dataset, blob_dist))

    return dataset


# sample a random PSD matrix with prescribed ratio of max_ev / min_ev
def random_PSD():
    diag = np.zeros((2, 2))
    diag[0, 0] = 3
    diag[1, 1] = diag[0, 0] * GMM_Params.ellpsoid_ratio
    random_unitary = ortho_group.rvs(2)
    covMat = random_unitary @ diag @ random_unitary.T
    return covMat


def main():
    # Generate ellipsoid blobs:
    # dataset = generate_dataset(GMM_Params.number_of_modes, GMM_Params.mean_to_variance_ratio)
    dataset = create_polygon_samples()
    # generate Gaussian mixture that approximates uniform distribution on dataset
    centers = np.zeros((GMM_Params.number_of_clusters, 2))
    gmm = mixture.GaussianMixture(n_components=GMM_Params.number_of_clusters, covariance_type='full').fit(dataset)

    # generate bounding box for each estimated model
    model_rect = []
    for idx in range(GMM_Params.number_of_clusters):
        model_rect.append(generate_bounding_rect(gmm.covariances_[idx], gmm.means_[idx]))

    print_srr(dataset, model_rect)


def print_srr(dataset, model_rect):
    fig, ax = plt.subplots()
    ax.scatter(dataset[:, 0], dataset[:, 1], s=50, cmap='viridis')
    for idx in range(GMM_Params.number_of_clusters):
        ax.add_patch(Rectangle(model_rect[idx].anchor,
                               width=model_rect[idx].width,
                               height=model_rect[idx].height,
                               angle=np.rad2deg(model_rect[idx].angle),
                               fill=False,
                               linewidth=2))
    plt.show()


def generate_bounding_rect(covariance, mean):
    # compute spectral decomposition
    eigenvalues, eigenvectors = np.linalg.eig(covariance)

    # shift both eigenvectors to positive orthant
    eigenvectors[:, 0] = shift_to_positive_orthant(eigenvectors[:, 0])
    eigenvectors[:, 1] = shift_to_positive_orthant(eigenvectors[:, 1])

    # compute rectangle attributes
    ellipsoid_angle = np.arctan2(-eigenvectors[0, 0], eigenvectors[1, 0])
    width = GMM_Params.side_width  # * np.sqrt(eigenvalues[1])
    height = GMM_Params.side_height  # * np.sqrt(eigenvalues[0])
    w_shift = width * np.array([eigenvectors[0, 1], eigenvectors[1, 1]])
    h_shift = height * np.array([eigenvectors[0, 0], eigenvectors[1, 0]])

    if np.sign(eigenvectors[0, 0]) != np.sign(eigenvectors[1, 0]):
        anchor = mean - w_shift + h_shift
    else:
        anchor = mean - w_shift - h_shift
    if ellipsoid_angle < -np.pi / 2:
        ellipsoid_angle += np.pi
    return BoundingRect(width=width, height=height, angle=ellipsoid_angle, anchor=anchor)


if __name__ == "__main__":
    main()

#######################  JUNK YARD #####################################
#
# def find_clusters(X, n_clusters, rseed=3):
#     # 1. Randomly choose clusters
#     rng = np.random.RandomState(rseed)
#     i = rng.permutation(X.shape[0])[:n_clusters]
#     centers = X[i]
#
#     while True:
#         # 2a. Assign labels based on closest center
#         labels = metrics.pairwise_distances_argmin(X, centers)
#
#         # 2b. Find new centers from means of points
#         new_centers = np.array([X[labels == i].mean(0)
#                                 for i in range(n_clusters)])
#
#         # 2c. Check for convergence
#         if np.all(centers == new_centers):
#             break
#         centers = new_centers
#
#     return centers, labels
