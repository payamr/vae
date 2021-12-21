import json
import os
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from datasets import load_mnist_images
from models import CVAE

OUTPUT_PATH = '/home/payam/Documents/vae_outputs'


def cluster_train_data(n_per_class: int = 2**16, batch_size=128):
    (train_images, train_labels), (_, _) = load_mnist_images(load_labels=True)
    datasets = {}
    for c in range(10):
        ind = np.where(train_labels == c)[0][:n_per_class]
        datasets[c] = tf.data.Dataset.from_tensor_slices(train_images[ind]).batch(batch_size)
    return datasets


def calculate_latents(
        clustered_datasets: Dict[int, tf.data.Dataset],
        model,
        cached_path: str = os.path.join(OUTPUT_PATH, 'train_latents.json'),
        force_recalculate: bool = False
):
    """calculate latent variables if not cached, else read them from disk"""
    if os.path.isfile(cached_path) and not force_recalculate:
        with open(cached_path, 'r') as fj:
            latents = json.load(fj)
        for c in latents.keys():
            for k in latents[c]:
                latents[c][k] = np.asarray(latents[c][k])
        return {int(c): v for c, v in latents.items()}

    latents = {}
    for c, dataset in clustered_datasets.items():
        latents[c] = {'mean': [], 'std': [], 'latent': []}
        for d in dataset:
            mean, logvar = model.encode(d)
            z = model.reparameterize(mean, logvar)
            latents[c]['mean'].append(mean)
            latents[c]['std'].append(np.exp(logvar)**0.5)
            latents[c]['latent'].append(z)
        for k in latents[c]:
            latents[c][k] = np.concatenate(latents[c][k], axis=0)

    with open(cached_path, 'w') as fj:
        latents_j = {c: {} for c in latents.keys()}
        for c, v in latents.items():
            for k in v:
                latents_j[c][k] = v[k].tolist()
        json.dump(latents_j, fj)

    return latents


clustered_data = cluster_train_data()

m = CVAE(latent_dim=2, input_shape=(28, 28, 1))
checkpoint_path = '/home/payam/Documents/log_dir/vae/weights-epoch-040'
m.load_weights(checkpoint_path)

lats = calculate_latents(clustered_data, m, force_recalculate=True)

fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12, 12), facecolor=(1, 1, 1))
for cl in range(10):
    ax[0, 0].scatter(lats[cl]['mean'][:, 0], lats[cl]['mean'][:, 1])
    ax[0, 0].set_title('latent means')
    ax[0, 1].scatter(lats[cl]['std'][:, 0], lats[cl]['std'][:, 1])
    ax[0, 1].set_title('latent std')
    ax[1, 0].scatter(lats[cl]['latent'][:, 0], lats[cl]['latent'][:, 1])
    ax[1, 0].set_title('latent vector')

# plt.show()
plt.savefig(os.path.join(OUTPUT_PATH, 'latent_scatter.png'))

min_latent = np.ones(2) * np.inf
max_latent = - np.ones(2) * np.inf
for cl in lats.keys():
    min_latent = np.minimum(min_latent, np.min(lats[cl]['latent'], axis=0))
    max_latent = np.maximum(max_latent, np.max(lats[cl]['latent'], axis=0))

print(min_latent)
print(max_latent)

grid_dim = 10
fig, ax = plt.subplots(nrows=grid_dim, ncols=grid_dim, figsize=(12, 12), facecolor=(1, 1, 1))
for i, col in enumerate(np.linspace(min_latent[0], max_latent[0], grid_dim)):
    for j, row in enumerate(np.linspace(min_latent[1], max_latent[1], grid_dim)):
        gen_im = m.decode(np.array([[col, row]]), apply_sigmoid=True)[0]
        ax[i, j].imshow(gen_im, cmap='Greys_r')
        ax[i, j].axis('off')

plt.tight_layout()
plt.axis('off')
# plt.show()
plt.savefig(os.path.join(OUTPUT_PATH, 'transition.png'))
