from matplotlib.patches import Ellipse
import GMM_Params
from BoundingRect import BoundingEllipse
from math_utils import shift_to_positive_orthant
import numpy as np
import matplotlib.pyplot as plt


def generate_bounding_ellpise(covariance, mean):

    # compute spectral decomposition
    eigenvalues, eigenvectors = np.linalg.eig(covariance)

    # shift both eigenvectors to positive orthant
    eigenvectors[:, 0] = shift_to_positive_orthant(eigenvectors[:, 0])
    eigenvectors[:, 1] = shift_to_positive_orthant(eigenvectors[:, 1])

    # compute rectangle attributes
    ellipsoid_angle = np.arctan2(-eigenvectors[0, 0], eigenvectors[1, 0])
    if ellipsoid_angle < -np.pi / 2:
        ellipsoid_angle += np.pi
    width = GMM_Params.ellpsoid_cover_ratio * np.sqrt(eigenvalues[1])
    height = GMM_Params.ellpsoid_cover_ratio * np.sqrt(eigenvalues[0])

    return BoundingEllipse(center=mean, width=width, height=height, angle=ellipsoid_angle)

    # w_shift = width * np.array([eigenvectors[0, 1], eigenvectors[1, 1]])
    # h_shift = height * np.array([eigenvectors[0, 0], eigenvectors[1, 0]])

    # if np.sign(eigenvectors[0, 0]) != np.sign(eigenvectors[1, 0]):
    #     anchor = mean - w_shift + h_shift
    # else:
    #     anchor = mean - w_shift - h_shift
    # return BoundingRect(width=width, height=height, angle=ellipsoid_angle, anchor=anchor)


def print_srr(dataset, model_rect):
    fig, ax = plt.subplots()
    ax.scatter(dataset[:, 0], dataset[:, 1], s=50, cmap='viridis')
    for idx in range(GMM_Params.number_of_clusters):
        ax.add_patch(Ellipse(model_rect[idx].center,
                             width=model_rect[idx].width,
                             height=model_rect[idx].height,
                             angle=np.rad2deg(model_rect[idx].angle),
                             fill=False,
                             linewidth=2))

