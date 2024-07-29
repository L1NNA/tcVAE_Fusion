import numpy as np
from tqdm import tqdm

import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from torch.utils.data import TensorDataset, DataLoader
import torch.distributions

from scipy.spatial import distance
from sklearn.metrics import accuracy_score, roc_auc_score

from loader import Dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def to_shape(a, shape):
    y_, x_ = shape
    y, x = a.shape
    y_pad = (y_-y)
    x_pad = (x_-x)
    return np.pad(a, ((y_pad//2, y_pad//2 + y_pad%2), (x_pad//2, x_pad//2 + x_pad%2)), mode='constant')

def get_neighbours(movie, target_movie, neuron_id, knn):
    """Returns the nearest neighbours of a neuron (distance calculated with the center of mass of each neuron"""
    cm = movie.centers[neuron_id]
    candidates = list()
    for candidate, candidate_cm in target_movie.centers.items():
        ed = distance.euclidean(cm, candidate_cm)
        candidates.append((ed, candidate))
    selected = [c[1] for c in sorted(candidates)[:knn]]
    return selected

class ConditionalVariationalEncoder(nn.Module):
    def __init__(self, latent_dims, size, condition_size):
        super(ConditionalVariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(size[0] * size[1] + condition_size, 512)
        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()      # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0     # KL loss

    def forward(self, x, c):
        x = torch.flatten(x, start_dim=1)
        x = torch.cat([x, c], 1)
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z

class ConditionalDecoder(nn.Module):
    def __init__(self, latent_dims, size, condition_size):
        super(ConditionalDecoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims + condition_size, 512)
        self.linear2 = nn.Linear(512, size[0] * size[1])
        self.size = size

    def forward(self, z, c):
        z = torch.cat([z, c], 1)
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 1, self.size[0], self.size[1]))

class ConditionalVariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims, size):
        super(ConditionalVariationalAutoencoder, self).__init__()
        self.gru = nn.GRU(1, latent_dims, batch_first=True)     # GRU used to get condition vector c for the VAE
        self.encoder = ConditionalVariationalEncoder(latent_dims, size, latent_dims)
        self.decoder = ConditionalDecoder(latent_dims, size, latent_dims)

    def forward(self, x, c):
        c = self.gru(c)[0][:,-1,:]        # Condition vector c
        z = self.encoder(x, c)
        return self.decoder(z, c)

    def encode(self, x, c):
        c = self.gru(c)[0][:,-1,:]        # Condition vector c
        return self.encoder(x, c)

def train(autoencoder, data, epochs=20):
    opt = torch.optim.Adam(autoencoder.parameters())
    bar = tqdm(total=len(data))
    for epoch in range(epochs):
        bar.reset()
        cumulative_loss = 0
        for ind, (x, t) in enumerate(data):
            x, t = x.to(device), t.unsqueeze(-1).to(device)     # To use the GPU
            opt.zero_grad()
            x_hat = autoencoder(x, t)
            loss = ((x - x_hat)**2).sum() + autoencoder.encoder.kl      # Reconstruction loss + KL loss
            cumulative_loss += loss
            loss.backward()
            opt.step()
            bar.set_description(f"Epoch {epoch+1}/{epochs} Loss {cumulative_loss/ind/len(x):.2f}")
            bar.update()
    return autoencoder

max_size_shape = (48, 48)
max_size_knn = (200, 200)
max_len = 500
latent_dims = 2

def get_temporal(movie, max_len=max_len):
    return [movie.get_temporal(i)[:max_len].tolist() for i in range(movie.total_neurons)]

def get_images(movie, max_size_shape=max_size_shape):
    neurons = [to_shape(movie.get_bbox_image(i), max_size_shape) for i in range(movie.total_neurons)]
    return neurons

def get_knn_images(movie, k=5, max_size_knn=max_size_knn):
    neurons = [to_shape(movie.get_bbox_multiple_image(i, k), max_size_knn) for i in range(movie.total_neurons)]
    return neurons

def encode_neuron_pairs(m1, m2):
    """Encoding function for neuron pairs using Variational Autoencoder."""

    m1_shape_images = get_images(m1)
    m2_shape_images = get_images(m2)

    m1_temporals = get_temporal(m1)

    m1_knn_images = get_knn_images(m1)
    m2_knn_images = get_knn_images(m2)

    # Use same temporal conditions for all images
    m1_conditions = torch.Tensor([m1_temporals[0]]*len(m1_shape_images)).unsqueeze(-1).to(device)
    m2_conditions = torch.Tensor([m1_temporals[0]]*len(m2_shape_images)).unsqueeze(-1).to(device)

    # Load saved models
    vae_shape = ConditionalVariationalAutoencoder(latent_dims, size=max_size_shape).to(device)
    vae_shape.load_state_dict(torch.load('../trained/vae_shape.pth'))
    vae_shape.eval()

    vae_neighbour = ConditionalVariationalAutoencoder(latent_dims, size=max_size_knn).to(device)
    vae_neighbour.load_state_dict(torch.load('../trained/vae_neighbour.pth'))
    vae_neighbour.eval()

    m1_shape = vae_shape.encode(torch.Tensor(np.array(m1_shape_images)).to(device), m1_conditions).cpu().detach().numpy()
    m2_shape = vae_shape.encode(torch.Tensor(np.array(m2_shape_images)).to(device), m2_conditions).cpu().detach().numpy()     # switch to m1 temporal

    m1_knn = vae_neighbour.encode(torch.Tensor(np.array(m1_knn_images)).to(device), m1_conditions).cpu().detach().numpy()
    m2_knn = vae_neighbour.encode(torch.Tensor(np.array(m2_knn_images)).to(device), m2_conditions).cpu().detach().numpy()     # switch to m1 temporal

    all_pairs = []
    all_features = []
    for i in range(m1.total_neurons):
        area = get_neighbours(m1, m2, i, knn=3)
        for j in area:
            all_features.append([
                1/(1+distance.euclidean(m1_shape[i], m2_shape[j])),
                1/(1+distance.euclidean(m1_knn[i], m2_knn[j])),
                1/(1+distance.euclidean(m1.centers[i], m2.centers[j])),
            ])
            all_pairs.append((i, j))
    return all_pairs, torch.Tensor(all_features)

class Fusion(nn.Module):
    def __init__(self, in_features, hidden=100, out_features=1):
        super(Fusion, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def fusionTrain(model, x, x_valid, y, y_valid, epochs=550):
    opt = torch.optim.Adam(model.parameters())
    bce = nn.BCELoss()
    bar = tqdm(total=epochs)
    for epoch in range(epochs):
        cumulative_loss = 0
        opt.zero_grad()

        loss = bce(model(x), y.unsqueeze(-1))
        cumulative_loss += loss
        loss.backward()
        opt.step()

        y_pred = model(x_valid).detach().numpy()
        acc = accuracy_score(y_valid, y_pred >= .5)
        sco = roc_auc_score(y_valid, y_pred)

        bar.set_description(f"Epoch {epoch+1}/{epochs} Loss {cumulative_loss:.4f} Acc {acc:.4f} {sco:.4f}")
        bar.update()
    return model

def training(ds):
    from pathlib import Path
    Path("../trained").mkdir(parents=True, exist_ok=True)

    # Loading the data for training
    training_match = ds.training_ds
    m1 = training_match.m1
    m2 = training_match.m2

    m1_shape_images = get_images(m1)
    m2_shape_images = get_images(m2)

    m1_temporals = get_temporal(m1)
    m2_temporals = get_temporal(m2)

    m1_knn_images = get_knn_images(m1)
    m2_knn_images = get_knn_images(m2)

    all_images = np.array(m1_shape_images + m2_shape_images)
    all_temporals = np.array(m1_temporals + m2_temporals)
    all_knn_images = np.array(m1_knn_images + m2_knn_images)

    dataloader_shape = DataLoader(TensorDataset(torch.Tensor(all_images), torch.Tensor(all_temporals)), batch_size=1)
    vae_shape = ConditionalVariationalAutoencoder(latent_dims, size=max_size_shape).to(device)
    vae_shape = train(vae_shape, dataloader_shape, epochs=30)

    dataloader_neighbour = DataLoader(TensorDataset(torch.Tensor(all_knn_images), torch.Tensor(all_temporals)), batch_size=1)
    vae_neighbour = ConditionalVariationalAutoencoder(latent_dims, size=max_size_knn).to(device)
    vae_neighbour = train(vae_neighbour, dataloader_neighbour, epochs=5)

    torch.save(vae_shape.state_dict(), '../trained/vae_shape.pth')
    torch.save(vae_neighbour.state_dict(), '../trained/vae_neighbour.pth')

    training_pairs, x = encode_neuron_pairs(m1, m2)
    y = torch.Tensor([1 if p in training_match.pairs else 0 for p in training_pairs])

    valid_match = ds.matches[1]
    valid_pairs, x_valid = encode_neuron_pairs(valid_match.m1, valid_match.m2)

    # Training the lightweight Fusion Neural Network
    y_valid = [1 if p in valid_match.pairs else 0 for p in valid_pairs]
    model = Fusion(len(x[0]))
    model = fusionTrain(model, x, x_valid, y, y_valid)

    torch.save(model.state_dict(), '../trained/fusion_model.pth')

class VAE():
    def __init__(self):
        self.name = 'VAE'

    def match(self, m1, m2):
        print('Preparing pairs...')

        test_pairs, x_test = encode_neuron_pairs(m1, m2)
        print('Infering')

        model = Fusion(len(x_test[0]))
        model.load_state_dict(torch.load('../trained/fusion_model.pth'))
        model.eval()

        y_pred = model(x_test).detach().numpy()

        similarity = np.zeros((m1.total_neurons, m2.total_neurons))

        for (i, j), score in zip(test_pairs, y_pred):
            similarity[i, j] = score[0]

        return similarity

if __name__ == "__main__":
    """Tests"""

    cases = {
        # Example: Keys = (m1, m2, groundtruth); value = (shift_x, shift_y)
    }

    ds = Dataset(cases=cases)

    # Training
    training(ds)

    # Evaluating
    ds.evaluate_all([VAE()])
