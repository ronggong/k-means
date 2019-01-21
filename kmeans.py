import numpy as np
import csv


def input_parser(fn_csv):
    """
    Parse the input.csv into python list
    Output:
        input_lines: [[x_0, y_0], [x_1, y_1], ...]
    """
    input_lines = []
    with open(fn_csv, 'r') as f:
        lines = csv.reader(f)
        for l in lines:
            input_lines.append([float(l[0]), float(l[1])])
    return np.array(input_lines)

def dis_euclidian(input_data_points, center):
    """
    Euclidian distance
    Input:
        input_data_points [[x_0, y_0], [x_1, y_1], ...]
        center [x_c, y_c]
    Output:
        euclidien distance [d_0, d_1, ...]
    """
    
    return np.sqrt(np.sum(np.power((input_data_points - center), 2.0), axis=1))

def error_kmeans_centers(centers_updated, centers_old):
    """
    Calculate the convergence error
    if the centers do not change.
    Input:
        centers_updated [[x_u_0, y_u_0], [x_u_1, y_u_1], ...]
        centers_old [[x_o_0, y_o_0], [x_o_1, y_o_1], ...]
    Output:
        err_km_center
    """
    return np.sum(np.abs(centers_updated-centers_old))

def error_kmeans(input_data_points, centers, k):
    """
    Calculate the k-means error. 
    The total error metric should also be accumulated 
    as the sum of the distances of each point to its assigned cluster centroid. 
    Input:
        input_data_points [[x_0, y_0], [x_1, y_1], ...]
        centers [[x_c_0, y_c_0], [x_c_1, x_c_1], ...]
    Output:
        err_km 
    """
    distance = np.zeros((input_array.shape[0], k))
    err = np.zeros(input_data_points.shape[0])
    for i in range(k):
        distance[:, i] = dis_euclidian(input_data_points, centers[i])
    clusters = np.argmin(distance, axis=1)
    for i in range(k):
        err[clusters==i] = distance[clusters==i, i]
    return np.sum(err), clusters


def kmeans(input_array, cluster_centers, k):
    """
    k-means EM algorithm.
    Expectation: each input data point is assigned to the closest cluster centroid measured using the Euclidian distance metric
    Maximisation: total error metric is minimised by adjusting the cluster centroids to the centre of the data points assigned to it.
    Input:
        input_array [[x_0, y_0], [x_1, y_1], ...]
        cluster_centers initialisation
    Output:
        cluster_centers
    """
    cluster_centers_old = np.zeros(cluster_centers.shape)
    distance = np.zeros((input_array.shape[0], k))

    err = 1.0
    while err != 0:
        # expectation
        for i in range(k):
            distance[:, i] = dis_euclidian(input_array, cluster_centers[i])
        clusters = np.argmin(distance, axis=1)

        # maximisation
        cluster_centers_old = np.copy(cluster_centers)
        for i in range(k):
            cluster_centers[i] = np.mean(input_array[clusters==i], axis=0)
        err = error_kmeans_centers(cluster_centers, cluster_centers_old)


if __name__ == "__main__":

    cluster_name = ['Adam', 'Bob', 'Charley', 'David', 'Edward']
    k = len(cluster_name) # cluster number
    cluster_centers = np.array([[-0.357, -0.253], [-0.055, 4.392], [2.674, -0.001], [1.044, -1.251], [-1.495, -0.090]])

    # parse input.csv
    input_array = input_parser('input.csv')
    # do k-means until convergence
    kmeans(input_array, cluster_centers, k)
    # calculate error metric
    err_km, clusters = error_kmeans(input_array, cluster_centers, k)
    # assign the cluster
    clusters = [cluster_name[c] for c in clusters]

    # write to OUTPUT.TXT
    with open('OUTPUT.TXT', 'w') as f:
        f.write("{:0.3f}".format(err_km)+'\n')
        for c in clusters:
            f.write(c+'\n')

    