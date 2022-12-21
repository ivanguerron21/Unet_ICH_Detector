from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import os
from models import UNet, UNet1
from tools import is_file, df_train_generator
from keras.callbacks import ModelCheckpoint
import pandas as pd


img_height = 256
img_width = 256
img_size = (img_height, img_width)
model_weights_name = 'unet1_weight_model.hdf5'  # change to the one you want to use as pretrained
test_num = len(os.listdir('test/img'))
BATCH_SIZE = 32  # 16
flag = False
train_data = pd.read_csv("df_even.csv")
skf = StratifiedKFold(n_splits=3, shuffle=True)
Y = train_data['No_Hemorrhage']


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

    learning_rate = 1e-4
    EPOCHS = 80

    for train_idx, val_idx in skf.split(train_data, Y):
        unet = UNet1(
            input_size=(img_width, img_height, 1),
            n_filters=32,
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

        # checkpoints/unet_normal if you are doing the other model
        filepath = 'checkpoints/unet_32/' + str(fold) + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
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
        histDF.to_csv('results/unet_32/historyCSV_fold_' + str(fold) + '.csv')
        # results/unet_normal if you are doing the other model

        unet.save_model('checkpoints/unet_32/' + str(fold) + 'unet_weight_model.hdf5')
        unet.save('checkpoints/unet_32/' + str(fold) + 'UNet1Test.h5')

        fold += 1
