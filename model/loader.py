import numpy as np
import os
from tqdm import tqdm

from scipy.io import loadmat
from scipy import stats, ndimage
from scipy.spatial import distance

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

from metrics import *

keys = ['A','A_prev','C','C_prev','C_raw','S','kernel','b','f','W','b0',
        'b0_new','options','P','Fs','file','frame_range','ids','tags',
        'Cn','PNR','Coor','neurons_per_patch','Df','C_df','S_df',
        'batches','file_id',]

def shift_image(X, dx, dy):
    """Shifts the image X, by dx and dy."""
    if dx > 0:
        X = np.pad(X, ((0, 0), (0, abs(dx))), mode='constant', constant_values=0)
    elif dx < 0:
        X = np.pad(X, ((0, 0), (abs(dx), 0)), mode='constant', constant_values=0)
    if dy > 0:
        X = np.pad(X, ((abs(dy), 0), (0, 0)), mode='constant', constant_values=0)
    elif dy < 0:
        X = np.pad(X, ((0, abs(dx)), (0, 0)), mode='constant', constant_values=0)

    X = np.roll(X, dx, axis=1)
    return X

class Movie:
    def __init__(self, mat_file, dx=None, dy=None):
        """Reads the Matlab file. dx and dy parameters are used to shift the movie."""

        self.dx = dx
        self.dy = dy

        self.data = dict()

        try:
            s_tuple = loadmat(mat_file)['neuron_struct'][0][0]  # [0][0] is used to retrieve the value of the saved structure name 's'
            for index, name in enumerate(s_tuple.dtype.fields):
                self.data[name] = s_tuple[index]

        except KeyError:
            s_tuple = loadmat(mat_file)
            for index, name in enumerate(s_tuple.keys()):
                if name[0:6] == "neuron":
                    new_name = name[6:]
                    self.data[new_name] = s_tuple[name]

        self.options = dict()
        for index, name in enumerate(self.data['options'][0][0].dtype.fields):
            self.options[name] = self.data['options'][0][0][index].tolist()

        self.total_neurons = self.data['A'].shape[1]        # Saves the spatial components of each neuron
        self.image_shape = self.data['b0_new'].transpose().shape
        self.centers = {i:self.get_center_of_mass(i) for i in range(self.total_neurons)}

    def print_data_shape(self):
        """Prints the shape of the data for each key."""
        for k, v in self.data.items():
            print(k, v.shape)

    def get_full_image(self):
        """Returns the full spatial image of neurons"""
        images = [self.get_spatial_image(i) for i in range(self.total_neurons)]
        return np.max(images, axis=0)

    def get_spatial_image(self, neuron_id, normalize=True):
        """Returns the correctly shifted spatial image of a specific neuron"""
        image = self.data['A'][:, neuron_id].todense().reshape(self.image_shape).transpose()
        if normalize:
            image = image/image.max()
        if self.dx is not None:
            image = shift_image(image, self.dx, self.dy)
        return image

    def get_spatial_multiple_image(self, neuron_id, knn, with_itself=True):
        """Returns the spatial image of a specific neuron's k-nearest neighbours, with or without itself"""
        neighbours = self.get_neighbours(neuron_id, knn)
        images = list()
        if with_itself:
            images.append(self.get_spatial_image(neuron_id))
        for j in range(knn):
            images.append(self.get_spatial_image(neighbours[j]))
        return np.max(images, axis=0)

    def get_bbox(self, neuron_id, return_image=False):
        """Returns the bounding box"""
        img = self.get_spatial_image(neuron_id)
        a = np.where(img != 0)
        bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
        if return_image:
            return img[bbox[0]:bbox[1], bbox[2]:bbox[3]]
        return bbox

    def get_bbox_image(self, neuron_id):
        """Returns the image of the bounding box"""
        return self.get_bbox(neuron_id, return_image=True)

    def get_bbox_multiple(self, neuron_id, knn, return_image=False, full_size_image=False):
        """Returns the bounding box of the k-nearest neighbours"""
        neighbours = self.get_neighbours(neuron_id, knn)
        b = np.zeros((knn, 4))
        for i in range(knn):
            img = self.get_spatial_image(neighbours[i])
            a = np.where(img != 0)
            b[i] = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
        bbox = np.min(b[:, 0]), np.max(b[:, 1]), np.min(b[:, 2]), np.max(b[:, 3])
        image = self.get_spatial_multiple_image(neuron_id, knn)
        if return_image:
            if full_size_image:
                return image
            else:
                return image[int(bbox[0]):int(bbox[1]), int(bbox[2]):int(bbox[3])]
        return bbox

    def get_bbox_multiple_image(self, neuron_id, knn, full_size_image=False):
        """Returns the image of the bounding box of the k-nearest neighbours"""
        return self.get_bbox_multiple(neuron_id, knn, return_image=True, full_size_image=full_size_image)

    def get_temporal(self, neuron_id):
        """Returns the temporal activity of a specific neuron"""
        return self.data['C'][neuron_id]

    def get_center_of_mass(self, neuron_id):
        """Returns the center of mass of the neuron"""
        cm = np.squeeze(np.asarray(self.get_spatial_image(neuron_id, normalize=False)))
        cm = ndimage.measurements.center_of_mass(cm)
        return cm

    def get_neighbours(self, neuron_id, knn):
        """Get the k-nearest neighbours of a specific neuron (distance calculated with the center of mass of each neuron)"""
        cm = self.centers[neuron_id]
        candidates = []

        for candidate, candidate_cm in self.centers.items():
            ed = distance.euclidean(cm, candidate_cm)
            candidates.append((ed, candidate))

        selected = [c[1] for c in sorted(candidates)[:knn]]
        return selected

    def plot_neuron(self, neuron_id):
        """Plot neuron"""
        plt.figure(figsize=(20, 20))
        gs = gridspec.Gridspec(1, 2)
        f1 = plt.subplot(gs[0, 0])
        plt.imshow(self.get_spatial_image(neuron_id), cmap='jet', interpolation='nearest')
        f1 = plt.subplot(gs[0, 1])
        plt.imshow(self.get_bbox_image(neuron_id), cmap='jet', interpolation='nearest')
        print(self.get_bbox(neuron_id))

    def plot_all(self, label_neuron_id='all'):
        """Plot all neurons"""
        plt.imshow(self.get_full_image(), cmap='jet', interpolation='nearest')
        if label_neuron_id == 'all':
            to_be_labeled = range(self.total_neurons)
        else:
            to_be_labeled = label_neuron_id
        for d in to_be_labeled:
            d_box = self.get_bbox(d)
            rect = patches.Rectangle((d_box[2], d_box[0]), d_box[3]-d_box[2], d_box[1]-d_box[0], linewidth=2, edgecolor='r', facecolor='none')
            ax = plt.gca()
            ax.add_patch(rect)
            ax.text(d_box[2], d_box[0], f'{d}', color='white', fontsize=15, weight='bold',)

class Match():
    """Class used for testing, to load ground truth data with the matching/corresponding neuron data"""
    def __init__(self, data_path, experiment, dx=None, dy=None):
        def2 = os.path.join(data_path, experiment[0])   # Data file 1
        hab = os.path.join(data_path, experiment[1])    # Data file 2, paired with data file 1
        csv = os.path.join(data_path, experiment[2])    # Ground truth file
        self.m1 = Movie(def2, dx, dy)
        self.m2 = Movie(hab)
        self.name = str(experiment[0][:4])
        self.pairs = list()

        # Encodes matching ground-truth in a matrix
        self.relevance_matrix = np.zeros((self.m1.total_neurons, self.m2.total_neurons))
        with open(csv) as rf:
            for l in rf.readlines():
                if l:
                    # d, h being two neuron IDs from m1 and m2 respectively, forming a neuron pair (from labelled data)
                    d, h = l.split(',')

                    # minus one because neuron ID starts from 1, but matrix index starts from 0
                    d = int(d)-1
                    h = int(h)-1
                    if h >= 0 and d >= 0:
                        if d >= self.m1.total_neurons:
                            print(f'error, {d} is larger than m1 {self.m1.total_neurons}')
                        elif h >= self.m2.total_neurons:
                            print(f'error, {h} is larger than m2 {self.m2.total_neurons}')
                        else:
                            self.pairs.append((d, h))
                            self.relevance_matrix[d, h] = 1

    def cal_pair_pearsonr(self, ind):
        """Returns and calculates the Pearson correlation coefficient and p-value for testing non-correlation of the temporal activity of d and h neurons (explained above)"""
        d, h = self.pairs[ind]

        #temporal d and h
        td = self.m1.get_temporal(d)
        th = self.m2.get_temporal(h)
        return stats.pearsonr(td[:len(th)], th[:len(td)])

    def plot_pair(self, ind):
        """Plots of a pair"""
        d, h = self.pairs[ind]

        plt.figure(figsize=(20, 20))
        gs = gridspec.GridSpec(1, 2)

        plt.subplot(gs[0, 0])
        self.m1.plot_all(label_neuron_id=d)
        plt.subplot(gs[0, 1])
        self.m2.plot_all(label_neuron_id=h)
        plt.show()

    def plot_time(self, ind):
        """Plots temporal activity of a pair"""
        d, h = self.pairs[ind]

        plt.figure(figsize=(20, 5))
        gs = gridspec.GridSpec(1, 2)

        td = self.m1.get_temporal(d)
        th = self.m2.get_temporal(h)

        plt.subplot(gs[0, 0])
        plt.plot(td)
        plt.subplot(gs[0, 1])
        plt.plot(th)
        plt.show()

def calculate_metric(similarity, relevance):
    """
    Calculates the metrics by comparing the similarity and relevance matrices
    Inputs:
    Similarity: Matrix of similarity values between i and j (similarity[i,j])
    Relevance: Matrix of 0/1 indicating if i is relevant to j
    """

    results = list()
    for s, r in tqdm(zip(similarity, relevance), total=len(relevance)):
        if np.sum(r) == 0:
            continue
        sorted_r = [t for _, _, t in sorted(zip(s, range(len(r)), r), reverse=True)]
        results.append(sorted_r)
    res = np.array(results)

    def _m(m, *args):
        """Helper function to calculate a metric"""
        return np.mean([m(r, *args) for r in res])

    return {
        'Mean Reciprocal Rank': mean_reciprocal_rank(res),
        'Precision': _m(r_precision),
        'Precision@1': _m(precision_at_k, 1),
        'Precision@5': _m(precision_at_k, 5),
        'Precision@10': _m(precision_at_k, 10),
        'Area Under PR Curve': _m(average_precision),
        'DCG@1': _m(dcg_at_k, 1),
        'DCG@5': _m(dcg_at_k, 5),
        'DCG@10': _m(dcg_at_k, 10),
        'NDCG@1': _m(ndcg_at_k, 1),
        'NDCG@5': _m(ndcg_at_k, 5),
        'NDCG@10': _m(ndcg_at_k, 10),
    }

metric_names = [
    'Mean Reciprocal Rank',
    'Precision',
    'Precision@1',
    'Precision@5',
    'Precision@10',
    'Area Under PR Curve',
    'DCG@1',
    'DCG@5',
    'DCG@10',
    'NDCG@1',
    'NDCG@5',
    'NDCG@10'
]

class Dataset:
    """Class to load specific training and testing dataset"""
    def __init__(self, cases={}, data_path='../training_data'):
        # cases is a dict of tuple keys (file1, file2, groundtruthfile) and tuple values, which is (None, None) by default.
        self.matches = list()
        for i, m in enumerate(cases.items()):
            dx, dy = m[1]
            print(f'loading {m[0]} {i+1}/{len(cases)}')
            self.matches.append(Match(data_path, m[0], dx, dy))

        # Used for training and testing, if not, reference self.matches
        self.training_ds = self.matches[0]
        self.testing_ds = self.matches[1:]

    def evaluate(self, matching_method):
        """Tests a matching method"""
        # A matching method takes two movies and outputs a similarity matrix
        result = dict()
        for test in self.testing_ds:
            print('evaluating', matching_method.name, 'on', test.name)
            similarity_matrix = matching_method.match(test.m1, test.m2)
            scores = calculate_metric(similarity_matrix, test.relevance_matrix)
            print(scores)
            result[test.name] = scores
        return result

    def evaluate_all(self, matching_methods, print_csv=True):
        """Tests all the given matching methods"""
        all_results = dict()
        for m in matching_methods:
            all_results[m.name] = self.evaluate(m)
        if print_csv:
            for test in self.testing_ds:
                print(test.name)
                for method in all_results:
                    row = [method]
                    for m_name in metric_names:
                        row.append(str(all_results[method][test.name][m_name]))
                    print(', '.join(row))
            print()
        return all_results
