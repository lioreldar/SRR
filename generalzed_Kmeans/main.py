import unittest

import matplotlib.pyplot as plt
import numpy as np
from sklearn import mixture
import GMM_Params
from display_utils import generate_bounding_ellpise, print_srr
from math_utils import generate_dataset, create_polygon_samples
import shapely.geometry as shapes


def main():
    # Generate ellipsoid blobs:

    #dataset = generate_dataset(GMM_Params.number_of_modes, GMM_Params.mean_to_variance_ratio)
    polygon = shapes.Polygon([(0, 0), (50, 0), (50, 50), (0, 0)])
    dataset = create_polygon_samples(polygon)

    # generate Gaussian mixture that approximates uniform distribution on dataset
    centers = np.zeros((GMM_Params.number_of_clusters, 2))
    gmm = mixture.GaussianMixture(n_components=GMM_Params.number_of_clusters, covariance_type='full').fit(dataset)

    # generate bounding box for each estimated model
    model_list = []
    for idx in range(GMM_Params.number_of_clusters):
        model_list.append(generate_bounding_ellpise(gmm.covariances_[idx], gmm.means_[idx]))

    print_srr(dataset, model_list)

    plt.show()


class SRR_unittests(unittest.TestCase):

    def test_polygon_cover(self):

        polygon = shapes.Polygon([(0, 0), (50, 0), (50, 50), (0, 0)])
        dataset = create_polygon_samples(polygon)

        # generate Gaussian mixture that approximates uniform distribution on dataset
        gmm = mixture.GaussianMixture(n_components=GMM_Params.number_of_clusters, covariance_type='full').fit(dataset)

        # generate bounding box for each estimated model
        model_list = []
        for idx in range(GMM_Params.number_of_clusters):
            model_list.append(generate_bounding_ellpise(gmm.covariances_[idx], gmm.means_[idx]))

        print_srr(dataset, model_list)

        plt.show()



    def test_Gaussian_mixture_cover(self):

        dataset = generate_dataset(GMM_Params.number_of_modes, GMM_Params.mean_to_variance_ratio)

        # generate Gaussian mixture that approximates uniform distribution on dataset
        gmm = mixture.GaussianMixture(n_components=GMM_Params.number_of_clusters, covariance_type='full').fit(dataset)

        # generate bounding box for each estimated model
        model_list = []
        for idx in range(GMM_Params.number_of_clusters):
            model_list.append(generate_bounding_ellpise(gmm.covariances_[idx], gmm.means_[idx]))

        print_srr(dataset, model_list)

        plt.show()







# ax.add_patch(Rectangle(model_rect[idx].anchor,
#                        width=model_rect[idx].width,
#                        height=model_rect[idx].height,
#                        angle=np.rad2deg(model_rect[idx].angle),
#                        fill=False,
#                        linewidth=2))


if __name__ == "__main__":
    main()
