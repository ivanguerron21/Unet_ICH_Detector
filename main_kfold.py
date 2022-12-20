from pathlib import Path

from sklearn.model_selection import StratifiedKFold
import os
import pandas as pd
from model_utils import dice_coef, iou, dice_coef_loss
from models import UNet, UNet1
from keras.optimizers import *
import matplotlib.pyplot as plt
from tools import train_generator, test_generator, save_results, is_file, prepare_dataset, show_image, df_train_generator
from keras.callbacks import ModelCheckpoint
import pandas as pd


img_height = 256
img_width = 256
img_size = (img_height, img_width)
train_path = 'train'
val_path = 'valid'
test_path = 'test'
save_path = Path('results')
version = 'base'
model_name = 'unet1_model.hdf5'
model_weights_name = 'unet1_weight_model.hdf5'
test_num = len(os.listdir('test/img'))
BATCH_SIZE = 32  # 16
flag = False
image_dir = "train_all"
train_data = pd.read_csv("df_even.csv")
skf = StratifiedKFold(n_splits=10, shuffle=True)
Y = train_data['No_Hemorrhage']


if __name__ == "__main__":

    def assign_image_fname(row):
        image_fname = str(int(row['Image']) + 1) + '.png'

        return image_fname


    train_generator_args = dict(rotation_range=0.2,
                                width_shift_range=0.05,
                                height_shift_range=0.05,
                                shear_range=0.05,
                                zoom_range=0.05,
                                horizontal_flip=True,
                                fill_mode='nearest')

    train_data['image_fname'] = train_data.apply(assign_image_fname, axis=1)

    fold = 1

    if is_file(file_name=model_weights_name) and flag:
        pretrained_weights = model_weights_name
    else:
        pretrained_weights = None

    learning_rate = 1e-4  # lo demas no sirve
    EPOCHS = 1000

    for train_idx, val_idx in skf.split(train_data, Y):
        unet = UNet1(
            input_size=(img_width, img_height, 1),
            n_filters=32,  # modificar
            pretrained_weights=pretrained_weights
        )
        unet.build(learning_rate=learning_rate, EPOCHS=EPOCHS)
        print(f"Currently training on Model N1, fold {fold}")
        training_data = train_data.iloc[train_idx]
        train_num = len(training_data)
        validation_data = train_data.iloc[val_idx]
        val_num = len(validation_data)
        steps_per_epoch = train_num // BATCH_SIZE
        steps_val = val_num // BATCH_SIZE
        train_gen = df_train_generator(
            aug_dict=train_generator_args,
            batch_size=BATCH_SIZE,
            dataframe=training_data,
            image_folder='train_all/img',
            mask_folder='train_all/mask',
            target_size=img_size
        )

        val_gen = df_train_generator(
            aug_dict=dict(),
            batch_size=BATCH_SIZE,
            dataframe=validation_data,
            image_folder='train_all/img',
            mask_folder='train_all/mask',
            target_size=img_size
        )

        filepath = str(fold) + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
        # creating a callback, hence best weights configurations will be saved
        callbacks = [ModelCheckpoint(filepath, monitor='val_loss', verbose=0,
                                     save_best_only=False, save_weights_only=False,
                                     mode='auto', period=100)]

        history = unet.fit(
            train_gen,
            epochs=EPOCHS,
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks,
            validation_data=val_gen,
            validation_steps=steps_val
        )

        histDF = pd.DataFrame.from_dict(history.history)
        histDF.to_csv('my_results/historyCSV_fold_' + str(fold) + '.csv')

        unet.save_model(str(fold) + 'unet_weight_model.hdf5')
        unet.save(str(fold) + 'UNet1Test.h5')

        fold += 1
