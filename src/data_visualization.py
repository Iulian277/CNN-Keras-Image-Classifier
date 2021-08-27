""" DATA VISUALIZATION """
from imports import *
from data_processing import train_batches

imgs, labels = next(train_batches)

# This function will plot images in the form of a grid with 1 row and 10 columns
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize = (20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Plot the images
print(labels)
plotImages(imgs)
