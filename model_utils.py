from keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, Activation, \
    MaxPooling2D, Dropout, concatenate, UpSampling2D, add
from keras.callbacks import ModelCheckpoint
from keras.layers.core import Lambda
import keras.backend as K


# defines input
def input_tensor(input_size):
    x = Input(input_size)
    return x


def single_conv(input_tensor, n_filters, kernel_size, data_format=None):
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size), data_format=data_format,
               activation='sigmoid')(input_tensor)
    return x


def double_conv(input_tensor, n_filters, kernel_size=3, batch_norm=False):
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), padding='same', kernel_initializer='he_normal')(input_tensor)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def triple_conv(input_tensor, n_filters, kernel_size=3, batch_norm=True):
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), padding='same',
               kernel_initializer='he_normal')(input_tensor)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), padding='same',
               kernel_initializer='he_normal')(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), padding='same',
               kernel_initializer='he_normal')(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def deconv(input_tensor, n_filters, kernel_size=2, stride=2):
    x = Conv2DTranspose(filters = n_filters, kernel_size = (kernel_size, kernel_size),
                        strides = (stride, stride), padding = 'same')(input_tensor)
    return x


def pooling(input_tensor, drop=False, dropout_rate=0.2, data_format=None):
    x = MaxPooling2D(pool_size=(2, 2), data_format=data_format, padding="same")(input_tensor)
    if drop:
        x = Dropout(rate=dropout_rate)(x)
    return x


def merge(input1, input2):
    x = concatenate([input1, input2])
    return x


def callback(name):
    return ModelCheckpoint(name, monitor='loss', verbose=1, save_best_only=True)


def up_and_concate(down_layer, layer, data_format='channels_first'):
    if data_format == 'channels_first':
        in_channel = down_layer.get_shape().as_list()[1]
    else:
        in_channel = down_layer.get_shape().as_list()[3]

    # up = Conv2DTranspose(out_channel, [2, 2], strides=[2, 2])(down_layer)
    up = UpSampling2D(size=(2, 2), data_format=data_format)(down_layer)

    if data_format == 'channels_first':
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
    else:
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))

    concate = my_concat([up, layer])

    return concate


# Recurrent Residual Convolutional Neural Network for future work
def rec_res_block(input_layer, out_n_filters, batch_normalization=False, kernel_size=[3, 3], stride=[1, 1],
                  padding='same', data_format='channels_first'):
    if data_format == 'channels_first':
        input_n_filters = input_layer.get_shape().as_list()[1]
    else:
        input_n_filters = input_layer.get_shape().as_list()[3]

    if out_n_filters != input_n_filters:
        skip_layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(
            input_layer)
    else:
        skip_layer = input_layer

    layer = skip_layer
    for j in range(2):

        for i in range(2):
            if i == 0:

                layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(
                    layer)
                if batch_normalization:
                    layer1 = BatchNormalization()(layer1)
                layer1 = Activation('relu')(layer1)
            layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(
                add([layer1, layer]))
            if batch_normalization:
                layer1 = BatchNormalization()(layer1)
            layer1 = Activation('relu')(layer1)
        layer = layer1

    out_layer = add([layer, skip_layer])
    return out_layer


def dice_coef(y_true, y_pred):
    smooth = 100
    y_truef = K.flatten(y_true)
    y_predf = K.flatten(y_pred)
    And = K.sum(y_truef * y_predf)
    return (2 * And + smooth) / (K.sum(y_truef) + K.sum(y_predf) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def iou(y_true, y_pred):
    smooth = 100
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac
