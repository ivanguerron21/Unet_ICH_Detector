from pathlib import Path
import os
import pandas as pd
from models import UNet, UNet1
import matplotlib.pyplot as plt
from tools import is_file, df_train_generator
from keras.callbacks import ModelCheckpoint

train_data = pd.read_csv("df_train.csv")
val_data = pd.read_csv("df_val.csv")

img_height = 256
img_width = 256
img_size = (img_height, img_width)
model_weights_name = 'unet_weight_model.hdf5'
train_num = len(os.listdir('train/img'))
val_num = len(os.listdir('valid/img'))
BATCH_SIZE = 32  # 16
flag = False

if __name__ == "__main__":

    checkpoints_path = Path('checkpoints')
    unet_32_path = checkpoints_path / 'unet_32'
    unet_normal_path = checkpoints_path / 'unet_normal'
    if not checkpoints_path.exists():
        checkpoints_path.mkdir()
        unet_32_path.mkdir()
        unet_normal_path.mkdir()

    results_path = Path('results')
    unet_32_path = results_path / 'unet_32'
    unet_normal_path = results_path / 'unet_normal'
    if not results_path.exists():
        results_path.mkdir()
        unet_32_path.mkdir()
        unet_normal_path.mkdir()

    def assign_image_fname(row):
        image_fname = str(int(row['Image']) + 1) + '.png'

        return image_fname


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
        n_filters=64,
        pretrained_weights=pretrained_weights
    )

    learning_rate = 1e-4
    EPOCHS = 1000
    unet.build(learning_rate=learning_rate, EPOCHS=EPOCHS)

    filepath = 'checkpoints/unet_normal/weights.{epoch:02d}-{val_loss:.2f}.hdf5'
    callbacks = [ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=100)]

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

    unet.save_model(model_weights_name)
    histDF = pd.DataFrame.from_dict(history.history)
    histDF.to_csv('results/unet_normal/historyCSV')
    unet.save('checkpoints/unet_normal/UNetTest.h5')

    plt.plot(history.history["loss"], label='Training')
    plt.plot(history.history["val_loss"], label='Validation')

    # Add in a title and axes labels
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Display the plot
    plt.legend(loc='best')
    plt.savefig("results/unet_normal/loss_fig.png")

    plt.clf()

    plt.plot(history.history["dice_coef"], label='Training')
    plt.plot(history.history["val_dice_coef"], label='Validation')

    # Add in a title and axes labels
    plt.title('Training and Validation DICE')
    plt.xlabel('Epochs')
    plt.ylabel('Dice')

    # Display the plot
    plt.legend(loc='best')
    plt.savefig("results/unet_normal/dice_fig.png")

    plt.clf()

    plt.plot(history.history["iou"], label='Training')
    plt.plot(history.history["val_iou"], label='Validation')

    # Add in a title and axes labels
    plt.title('Training and Validation IOU')
    plt.xlabel('Epochs')
    plt.ylabel('IOU')

    # Display the plot
    plt.legend(loc='best')
    plt.savefig("results/unet_normal/iou_fig.png")
