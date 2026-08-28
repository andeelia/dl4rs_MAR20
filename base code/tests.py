import os.path

import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import torch

if __name__ == '__main__':
    a = torch.randn((1, 16, 16))

    layer = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7, padding=0, padding_mode="replicate")

    print(layer(a).shape)

    #BASE_PATH = "/home/caipi/PycharmProjects/PythonProject1/"

    #img = io.imread(os.path.join(BASE_PATH, "data/true_color_images/train/RGB_ar037_2019_n_06_04_0.png"))

    #print(img.shape)

    #img_definition = np.transpose(img, (2, 0, 1))

    #greens = img_definition[1, :, :]

    #plt.imshow(greens, cmap="Greens")
    #plt.colorbar()
    #plt.show()

