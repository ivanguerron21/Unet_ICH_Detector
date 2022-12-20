from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import random
import skimage.io as io
import skimage.transform as trans
import numpy as np
from PIL import Image
from pathlib import Path


def prepare_dataset(
    path_to_data,
    image_folder,
    mask_folder,
    n_samples,
    as_gray=True
):
    """ Prepare Dataset
    Function that takes path to DataSet folder
    which has image and mask folder
    Each image and mask are transformed to square formats:
    reads both image and mask, creates new image and mask;
    generates random spacing coefficient,
    adds original image and paddings to them to make them square,
    then saves new masks and images and delets originals
    """
    path_to_image = os.path.join(path_to_data, image_folder)
    path_to_mask = os.path.join(path_to_data, mask_folder)

    for i in range(1, n_samples + 1):
        try:
            img_name = os.path.join(path_to_image,"%d.png" % i)
            mask_name = os.path.join(path_to_mask, "%d.png" % i)

            coefficient = random.uniform(0, 2)

            img = io.imread(fname = img_name, as_gray = as_gray)
            os.remove(img_name)
            new_img = square_image(img, random = coefficient)
            new_img = (new_img * 255).astype('uint8')
            io.imsave(fname = img_name, arr = new_img)

            mask = io.imread(fname = mask_name,as_gray = as_gray)
            os.remove(mask_name)
            new_mask = square_image(mask, random = coefficient)
            new_mask = (new_mask * 255).astype('uint8')
            io.imsave(fname = mask_name, arr = new_mask)

            print("Successfully added paddings to image and mask #%d" % i)
        except:
            print("Adding paddings failed at #%d" % i)

    print("All images and masks were resized to SQUARE format")


def df_train_generator(
    aug_dict,
    batch_size,
    dataframe,
    image_folder,
    mask_folder,
    target_size,
    image_color_mode='grayscale',
    mask_color_mode='grayscale'
):
    """ Image Data Generator
    Function that generates batches of data (img, mask) for training
    from specified folder. Returns images with specified pixel size
    Does preprocessing (normalization to 0-1)
    """
    # no augmentation, only rescaling
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_dataframe(
        dataframe,
        directory=image_folder,
        x_col="image_fname",
        y_col="No_Hemorrhage",
        class_mode=None,
        shuffle=True,
        target_size=target_size,
        batch_size=batch_size,
        color_mode=image_color_mode,
        seed=1
    )
    mask_generator = mask_datagen.flow_from_dataframe(
        dataframe,
        directory=mask_folder,
        x_col="image_fname",
        y_col="No_Hemorrhage",
        class_mode=None,
        shuffle=True,
        target_size=target_size,
        batch_size=batch_size,
        color_mode=mask_color_mode,
        seed=1
    )

    train_g = zip(image_generator, mask_generator)
    for (img, mask) in train_g:
        img, mask = adjust_data(img, mask)
        yield img, mask


def train_generator(
    aug_dict,
    batch_size,
    train_path,
    image_folder,
    mask_folder,
    target_size,
    image_color_mode='grayscale',
    mask_color_mode='grayscale'
):
    """ Image Data Generator
    Function that generates batches of data (img, mask) for training
    from specified folder. Returns images with specified pixel size
    Does preprocessing (normalization to 0-1)
    """
    # no augmentation, only rescaling
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        seed=1
    )
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        seed=1
    )
    train_g = zip(image_generator, mask_generator)
    for (img, mask) in train_g:
        img, mask = adjust_data(img, mask)
        yield img, mask


def adjust_data(img, mask):
    img = img / 255
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return (img, mask)


def test_generator(
    test_path,
    target_size,
    as_gray=True
):
    """ Image Data Generator
    Function that generates batches od data for testing from specified folder
    Reads images as grey, makes them square, scales them
    Returns images with specified pixel size
    Does preprocessing (normalization to 0-1)
    """
    l = os.listdir(test_path)
    for i in l:
        img = io.imread(os.path.join(test_path, i), as_gray=as_gray)
        img = square_image(img)
        img = reshape_image(img, target_size)
        yield img


def save_results(
    npyfile,
):
    # Overlap
    Img = []
    Msk = []
    new_images = []
    predicted_names = []
    names = []
    l_img = os.listdir("test/img")
    l_msk = os.listdir("test/mask")

    for img in l_img:
        img1 = Image.open("test/img/" + img)
        names.append(img)
        img1 = img1.copy()
        Img.append(img1)
    for msk in l_msk:
        msk1 = Image.open("test/mask/" + msk)
        msk1 = msk1.copy()
        Msk.append(msk1)

    cont = 0
    predicted_path = Path("predicted_masks")
    first_overlap_path = predicted_path / "first_overlap"
    only_predicted_path = predicted_path / "only_predicted"
    final_path = predicted_path / "final"

    if not predicted_path.exists():
        predicted_path.mkdir()
        first_overlap_path.mkdir()
        only_predicted_path.mkdir()
        final_path.mkdir()
    for img, msk, name in zip(Img, Msk, names):
        new_img = Image.blend(img, msk, 0.5)
        new_images.append(new_img)
        new_img.save(first_overlap_path / name)
        cont += 1

    for i, (item, name) in enumerate(zip(npyfile, l_img)):
        img = normalize_mask(item)
        img = (img * 255).astype('uint8')
        name = f'{l_img[i].strip(".png")}_predict.png'
        predicted_names.append(name)
        io.imsave(os.path.join(only_predicted_path, name), img)

    for img, name in zip(Img, predicted_names):
        predicted_mask = Image.open(only_predicted_path / name).convert("RGBA")
        data = np.array(predicted_mask)
        red, green, blue, alpha = data.T

        white_areas = (red == 255) & (blue == 255) & (green == 255)
        data[..., :-1][white_areas.T] = (255, 0, 0)

        im2 = Image.fromarray(data)
        new_img = Image.blend(img, im2, 0.6)
        new_img.save(final_path / name)


def is_file(
    file_name
) -> bool:
    """ Is File
    Check if file exists
    Later used to check if user has pretrained models
    """
    return os.path.isfile(file_name)


def square_image(img, random=None):
    """ Square Image
    Function that takes an image (ndarray),
    gets its maximum dimension,
    creates a black square canvas of max dimension
    and puts the original image into the
    black canvas's center
    If random [0, 2] is specified, the original image is placed
    in the new image depending on the coefficient,
    where 0 - constrained to the left/up anchor,
    2 - constrained to the right/bottom anchor
    """
    size = max(img.shape[0], img.shape[1])
    new_img = np.zeros((size, size),np.float32)
    ax, ay = (size - img.shape[1])//2, (size - img.shape[0])//2

    if random and not ax == 0:
        ax = int(ax * random)
    elif random and not ay == 0:
        ay = int(ay * random)

    new_img[ay:img.shape[0] + ay, ax:ax+img.shape[1]] = img
    return new_img


def reshape_image(img, target_size):
    """ Reshape Image
    Function that takes an image
    and rescales it to target_size
    """
    img = trans.resize(img, target_size)
    img = np.reshape(img, img.shape+(1,))
    img = np.reshape(img, (1,)+img.shape)
    return img


def normalize_mask(mask):
    """ Mask Normalization
    Function that returns normalized mask
    Each pixel is either 0 or 1
    """
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return mask


def show_image(img):
    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()
