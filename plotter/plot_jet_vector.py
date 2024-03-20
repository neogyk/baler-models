import vector
import numpy as np
from datasets import Dataset
from vector import Vector, VectorObject4D, Lorentz
from matplotlib import pyplot as plt
from matplotlib import cm as CM
plt.style.use('ggplot')


input_path = "top_tag_parquet/train/data-00000-of-00008.arrow"
output_path = "baler/example/output/decompressed_output/"

dataset = Dataset.from_file(input_path)
dataset.set_format(type='numpy')

features = list(dataset.features.keys())
data = [dataset[feature] for feature in features]
data = np.array(data).T

data_decompressed = np.load(output_path + "decompressed.npz")["data"][:, :, 0]


def make_plot(data, i):
    """_summary_

    Args:
        data (_type_): _description_
        i (_type_): _description_
    """
    vectors = [vector.VectorObject4D(px=data[i, 1+v*4], py=data[i, 2+v*4], pz=data[i, 3+v*4], E=data[i, 0+v*4]) for v in range(0, 199)]
    eta_array = np.array([v.eta for v in vectors])
    phi_array = np.array([v.phi for v in vectors])

    Z, xedges, yedges = np.histogram2d(eta_array, phi_array, bins=10)

    plt.pcolormesh(xedges, yedges, Z.T, cmap=CM.terrain_r)
    plt.ylabel("eta")
    plt.xlabel("phi")
    plt.colorbar()

if __name__ == "__main__":
    make_plot(data_decompressed, 100)
