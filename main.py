import os
import pandas as pd
from model_utils import dice_coef, iou, dice_coef_loss
from models import UNet, UNet1
from keras.optimizers import *
import matplotlib.pyplot as plt
from tools import train_generator, test_generator, save_results, is_file, prepare_dataset, show_image, df_train_generator
from keras.callbacks import ModelCheckpoint

train_data = pd.read_csv("df_train_even.csv")
val_data = pd.read_csv("df_val_even.csv")

img_height = 256
img_width = 256
img_size = (img_height, img_width)
train_path = 'train'
val_path = 'valid'
test_path = 'test'
save_path = 'results'
version = 'base'
model_name = 'unet_model.hdf5'
model_weights_name = 'unet_weight_model.hdf5'
train_num = len(os.listdir('train/img'))
val_num = len(os.listdir('valid/img'))
test_num = len(os.listdir('test/img'))
BATCH_SIZE = 32  # 16
flag = False

if __name__ == "__main__":
    def assign_image_fname(row):
        image_fname = str(int(row['Image']) + 1) + '.png'

        return image_fname


    # create a new column with image file names
    train_data['image_fname'] = train_data.apply(assign_image_fname, axis=1)
    val_data['image_fname'] = val_data.apply(assign_image_fname, axis=1)

    train_generator_args = dict(rotation_range=0.2,
                                width_shift_range=0.05,
                                height_shift_range=0.05,
                                shear_range=0.05,
                                zoom_range=0.05,
                                horizontal_flip=True,
                                fill_mode='nearest')

    # generates training set

    train_gen = df_train_generator(
        aug_dict=train_generator_args,
        batch_size=BATCH_SIZE,
        dataframe=train_data,
        image_folder='train/img',
        mask_folder='train/mask',
        target_size=img_size
    )

    val_gen = df_train_generator(
        aug_dict=dict(),
        batch_size=BATCH_SIZE,
        dataframe=val_data,
        image_folder='valid/img',
        mask_folder='valid/mask',
        target_size=img_size
    )

    # train_gen = train_generator(
    #     aug_dict=train_generator_args,
    #     batch_size=BATCH_SIZE,
    #     train_path=train_path,
    #     image_folder='img',
    #     mask_folder='mask',
    #     target_size=img_size
    # )
    #
    # val_gen = train_generator(
    #     aug_dict=dict(),
    #     batch_size=BATCH_SIZE,
    #     train_path=val_path,
    #     image_folder='img',
    #     mask_folder='mask',
    #     target_size=img_size
    # )

    # check if pretrained weights are defined
    if is_file(file_name=model_weights_name) and flag:
        pretrained_weights = model_weights_name
    else:
        pretrained_weights = None

    # build model
    unet = UNet(
        input_size=(img_width, img_height, 1),
        n_filters=64,  # modificar
        pretrained_weights=pretrained_weights
    )

    learning_rate = 1e-4  # lo demas no sirve
    EPOCHS = 1000
    unet.build(learning_rate=learning_rate, EPOCHS=EPOCHS)

    filepath = 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
    # creating a callback, hence best weights configurations will be saved
    callbacks = [ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=200)]

    steps_per_epoch = train_num // BATCH_SIZE
    steps_val = val_num // BATCH_SIZE

    history = unet.fit(
        train_gen,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        validation_data=val_gen,
        validation_steps=steps_val
    )

    # # saving model weights
    unet.save_model(model_weights_name)
    histDF = pd.DataFrame.from_dict(history.history)
    histDF.to_csv('historyCSV')
    unet.save('UNet1Test.h5')

    plt.plot(history.history["loss"], label='Training')
    plt.plot(history.history["val_loss"], label='Validation')

    # Add in a title and axes labels
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Display the plot
    plt.legend(loc='best')
    plt.savefig("loss_fig.png")

    plt.clf()

    plt.plot(history.history["dice_coef"], label='Training')
    plt.plot(history.history["val_dice_coef"], label='Validation')

    # Add in a title and axes labels
    plt.title('Training and Validation DICE')
    plt.xlabel('Epochs')
    plt.ylabel('Dice')

    # Display the plot
    plt.legend(loc='best')
    plt.savefig("dice_fig.png")

    plt.clf()

    plt.plot(history.history["iou"], label='Training')
    plt.plot(history.history["val_iou"], label='Validation')

    # Add in a title and axes labels
    plt.title('Training and Validation IOU')
    plt.xlabel('Epochs')
    plt.ylabel('IOU')

    # Display the plot
    plt.legend(loc='best')
    plt.savefig("iou_fig.png")

    # # generated testing set
    # test_gen = test_generator(test_path, img_size)
    #
    # # display results
    # results = unet.predict_generator(test_gen, 30, verbose=1)
    # save_results(save_path, test_path, results)
