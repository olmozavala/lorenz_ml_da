import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from os import path, makedirs
from os.path import join

seed = 132
np.random.seed(seed)

HERE = path.dirname(path.abspath(__file__))
REPO_ROOT = path.abspath(path.join(HERE, ".."))
cache_dir = join(REPO_ROOT, "dataset_cache")

# Load all npz files in the cache directory
npz_files = glob(join(cache_dir, '*.npz'))

if len(npz_files) == 0:
    raise FileNotFoundError(
        f"No .npz files found in {cache_dir!r}. "
        "Generate/cache datasets first, or point this script at the correct cache directory."
    )

# random select up to 3 files
sample_n = min(3, len(npz_files))
npz_files_sample = np.random.choice(npz_files, sample_n, replace=False)

ds = []
ts = []
for file in npz_files_sample:
    data = np.load(file)
    d = data['d']
    t = data['t']
    ds.append(d)
    ts.append(t)


fig = plt.figure(figsize=(30, 10))
axs = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(sample_n)]
for i in range(sample_n):

    # add starting point
    axs[i].scatter(ds[i][0, 0], ds[i][0, 1], ds[i][0, 2], color='red', s=100)

    # add trajectory
    axs[i].plot(ds[i][:, 0], ds[i][:, 1], ds[i][:, 2])

    axs[i].set_title(f'Dataset {i}, [x,y,z] = [{ds[i][0, 0]:.2f}, {ds[i][0, 1]:.2f}, {ds[i][0, 2]:.2f}]')
    axs[i].set_xlabel('X')
    axs[i].set_ylabel('Y')
    axs[i].set_zlabel('Z')

    print(f'Dataset {i}, mean [x,y,z] = [{np.mean(ds[i][:, 0]):.2f}, {np.mean(ds[i][:, 1]):.2f}, {np.mean(ds[i][:, 2]):.2f}]')
    print(f'Dataset {i}, std [x,y,z] = [{np.std(ds[i][:, 0]):.2f}, {np.std(ds[i][:, 1]):.2f}, {np.std(ds[i][:, 2]):.2f}]')

evidence_dir = join(REPO_ROOT, "evidence")
makedirs(evidence_dir, exist_ok=True)
plt.savefig(join(evidence_dir, "check_attractor.png"))

## Check chaos

sep1 = np.linalg.norm(ds[0] - ds[1], axis=1)
sep2 = np.linalg.norm(ds[1] - ds[2], axis=1)
sep3 = np.linalg.norm(ds[2] - ds[0], axis=1)

fig, axs = plt.subplots(1, 3, figsize=(30, 10))
axs[0].semilogy(sep1, label='Trajectory 1 vs 2')
axs[1].semilogy(sep2, label='Trajectory 2 vs 3')
axs[2].semilogy(sep3, label='Trajectory 3 vs 1')
axs[0].set_xlabel('steps')
axs[0].set_ylabel('||δx|| (Euclidean distance)')
axs[0].set_title('Should grow exponentially early on')
axs[0].legend()
axs[1].legend()
axs[2].legend()
plt.savefig(join(evidence_dir, "check_divergence.png"), dpi=150)
plt.show()