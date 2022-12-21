import os
from models import UNet, UNet1
from keras.optimizers import *
from tools import train_generator, test_generator, save_results

img_height = 256
img_width = 256
img_size = (img_height, img_width)
BATCH_SIZE = 1
test_path = "test"
test_path_img = "test/img"
test_num = len(os.listdir('test/img'))

if __name__ == "__main__":
    unet = UNet1(
        input_size=(img_width, img_height, 1),
        n_filters=32,
        pretrained_weights="checkpoints/unet_32/10weights.800-0.10.hdf5"
    )

    learning_rate = 1e-4  # lo demas no sirve
    EPOCHS = 1000
    unet.build(learning_rate=learning_rate, EPOCHS=EPOCHS)

    test_gen = train_generator(
        aug_dict=dict(),
        batch_size=BATCH_SIZE,
        train_path=test_path,
        image_folder='img',
        mask_folder='mask',
        target_size=img_size
    )

    steps = test_num//BATCH_SIZE
    unet.evaluate(test_gen, steps=steps)
    test_gen = test_generator(test_path_img, img_size)
    results = unet.predict_generator(test_gen, test_num, verbose=1)
    save_results(results)
