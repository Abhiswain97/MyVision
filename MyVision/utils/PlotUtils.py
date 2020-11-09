import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np


def show_batch(dataset, batch_size=5, classes=[]):
    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis("off")
        plt.show()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # get some random training images
    dataiter = iter(loader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(" ".join("%5s" % classes[labels[j]] for j in range(batch_size)))