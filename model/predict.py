import numpy as np
import os
from loader import *
from registration_vae import VAE
from time import perf_counter

from pathlib import Path
Path("../results").mkdir(parents=True, exist_ok=True)

class Pairs:
    """Creates a Movie pair for inference"""
    def __init__(self, m1, m2, name, dx=None, dy=None):
        self.m1 = Movie(m1, dx, dy)
        self.m2 = Movie(m2)
        self.name = name
        self.pairs = np.zeros((self.m1.total_neurons, 3))       # 3 columns: m1 neurons, m2 neurons, similarity

    def get_pairs(self, matching_method, dynamic_threshold=False):
        print('evaluating', matching_method.name, 'on', self.name)
        similarity_matrix = matching_method.match(self.m1, self.m2)
        argsorted = np.argsort(similarity_matrix)
        for i in range(similarity_matrix.shape[0]):
            if np.nonzero(similarity_matrix[i])[0].size == 0:
                self.pairs[i, 0] = i + 1
                self.pairs[i, 1] = -1
            else:
                while self.pairs[i, 0] == 0:
                    num_idx = i     #current row for biggest similarity
                    num_idy = argsorted[i][-1]      # current column for biggest similarity
                    num = similarity_matrix[i][num_idy] #current biggest similarity
                    if num == 0 or self.pairs[i, 1] == -1:
                        self.pairs[i, 0] = i + 1
                        self.pairs[i, 1] = -1
                        break

                    ls = argsorted[i+1:, -1]
                    for j in range(len(ls)):
                        if ls[j] == num_idy and self.pairs[i+j+1, 1] == 0:
                            if similarity_matrix[i+j+1][num_idy] > num:
                                similarity_matrix[num_idx][num_idy] = 0
                                argsorted = np.argsort(similarity_matrix)
                                num = similarity_matrix[i+j+1][num_idy]
                                num_idx = i+j+1
                                # num_idy stays the same
                            else:
                                similarity_matrix[i+j+1][num_idy] = 0
                                argsorted = np.argsort(similarity_matrix)
                    similarity_matrix[:, num_idy] = np.zeros(similarity_matrix.shape[0])
                    self.pairs[num_idx, 0] = num_idx + 1
                    self.pairs[num_idx, 1] = num_idy + 1
                    self.pairs[num_idx, 2] = num

        self.pairs = self.pairs[:, :2].astype(int)
        if dynamic_threshold:
            temp = np.where(self.pairs[:, -1]!=0, self.pairs[:, -1], np.nan)
            mu = np.nanmean(temp)
            sigma = np.nanstd(temp)
            theta = mu - 2 * sigma
            for i in range(len(self.pairs)):
                if self.pairs[i, -1] < theta:
                    self.pairs[i, 1] = -1       # similarity too low under threshold theta
        # Check
        assert self.check_duplicate()
        return self.pairs

    def check_duplicate(self):
        sort, count = np.unique(self.pairs[:, 1], return_counts=True)
        count[0] = 1        # Removes the duplicated -1
        count = np.where(count == 1, True, False)
        assert np.all(count)
        return np.all(count)

def test(data_dir: str, results_dir: str):
    """Inference function for VAE model, also calculates performance time. This evaluates, in sequential (alphabetical) order, each possible movie pair. Thus, for n movies, there are (n - 1)! inferences to be done."""
    start = perf_counter()
    files = list()
    for file in sorted(os.listdir(data_dir)):
        if os.path.isfile(os.path.join(data_dir, file)):
            files.append(file)
    print(f"Files in {os.path.basename(data_dir)}: {files}")

    done = list()
    x_points = list()
    y_points = list()
    for i in range(len(files)):
        done.append(i)
        for j in range(len(files)):
            if j not in done:
                step_start = perf_counter()
                name = f'{Path(files[i]).stem}-{Path(files[j]).stem}'
                m1 = os.path.join(data_dir, files[i])
                m2 = os.path.join(data_dir, files[j])
                all_pairs = Pairs(m1, m2, name)
                x_points.append(all_pairs.m1.total_neurons)

                all_pairs = all_pairs.get_pairs(VAE())          # Infering pairs with VAE model

                np.savetxt(results_dir + f'/{name}.csv', all_pairs, fmt='%d', delimiter=',')
                step_end = perf_counter()
                step_performance = step_end - step_start
                y_points.append(step_performance)
                print(f"Time for this movie pair: {step_performance} for {x_points[-1]} neurons\n")
    end = perf_counter()
    performance = end - start
    step_performance = list(zip(x_points, y_points))
    print("Time for this directory:", performance, "\n")
    return performance, step_performance

if __name__ == "__main__":
    for directory in sorted(os.listdir("../data")):
        results_dir = f"../results/{directory}"
        data_dir = f"../data/{directory}"
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        test(data_dir, results_dir)
