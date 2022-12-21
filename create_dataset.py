# from split_raw_data import do_split
from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import Image
from pathlib import Path
import os


def normal_train_test_split():
    num_hem = 318
    num_no_hem = 2496
    df_hem = pd.read_csv("hemorrhage_diagnosis_raw_ct.csv")
    needed = num_no_hem - num_hem
    df = df_hem.drop(df_hem[df_hem['No_Hemorrhage'] == 1].sample(n=needed).index)
    df.to_csv("new_hemorrhage_diagnosis.csv")

    df = pd.read_csv("new_hemorrhage_diagnosis.csv")
    df.rename(columns={'Unnamed: 0': 'Image'}, inplace=True)

    df_train, df_test_1 = train_test_split(df, test_size=0.20, random_state=107, stratify=df["No_Hemorrhage"])
    df_val, df_test = train_test_split(df_test_1, test_size=0.50, random_state=107
                                       , stratify=df_test_1["No_Hemorrhage"])

    df_train.to_csv("df_train.csv")
    df_val.to_csv("df_val.csv")
    df_test.to_csv("df_test.csv")

    df_train = pd.read_csv("df_train.csv")
    df_val = pd.read_csv("df_val.csv")
    df_test = pd.read_csv("df_test.csv")

    train_path = Path('train')
    train_img_path = train_path / 'img'
    train_mask_path = train_path / 'mask'
    if not train_path.exists():
        train_path.mkdir()
        train_img_path.mkdir()
        train_mask_path.mkdir()

    val_path = Path('valid')
    val_img_path = val_path / 'img'
    val_mask_path = val_path / 'mask'
    if not val_path.exists():
        val_path.mkdir()
        val_img_path.mkdir()
        val_mask_path.mkdir()

    test_path = Path('test')
    test_img_path = test_path / 'img'
    test_mask_path = test_path / 'mask'
    if not test_path.exists():
        test_path.mkdir()
        test_img_path.mkdir()
        test_mask_path.mkdir()

    for row in df_train.iterrows():
        pic_num = str(int(row[1][0]) + 1)
        img = Image.open(f'data/image/' + pic_num + '.png')
        img.save(f'train/img/' + pic_num + '.png')
        msk = Image.open(f'data/label/' + pic_num + '.png')
        msk.save(f'train/mask/' + pic_num + '.png')

    for row in df_val.iterrows():
        pic_num = str(int(row[1][0]) + 1)
        img = Image.open(f'data/image/' + pic_num + '.png')
        img.save(f'valid/img/' + pic_num + '.png')
        msk = Image.open(f'data/label/' + pic_num + '.png')
        msk.save(f'valid/mask/' + pic_num + '.png')

    for row in df_test.iterrows():
        pic_num = str(int(row[1][0]) + 1)
        img = Image.open(f'data/image/' + pic_num + '.png')
        img.save(f'test/img/' + pic_num + '.png')
        msk = Image.open(f'data/label/' + pic_num + '.png')
        msk.save(f'test/mask/' + pic_num + '.png')


def data_set_k_fold():
    num_hem = 318
    num_no_hem = 2496
    df_hem = pd.read_csv("hemorrhage_diagnosis_raw_ct.csv")
    needed = num_no_hem-num_hem
    df = df_hem.drop(df_hem[df_hem['No_Hemorrhage'] == 1].sample(n=needed).index)
    df.to_csv("new_hemorrhage_diagnosis.csv")

    df = pd.read_csv("new_hemorrhage_diagnosis.csv")
    df.rename(columns={'Unnamed: 0': 'Image'}, inplace=True)

    df_train, df_test = train_test_split(df, test_size=0.10, random_state=107, stratify=df["No_Hemorrhage"])

    df_train.to_csv("df_even.csv")
    df_test.to_csv("df_test_even.csv")

    df_train = pd.read_csv("df_even.csv")
    df_test = pd.read_csv("df_test_even.csv")

    train_path = Path('train_all')
    train_img_path = train_path / 'img'
    train_mask_path = train_path / 'mask'
    if not train_path.exists():
        train_path.mkdir()
        train_img_path.mkdir()
        train_mask_path.mkdir()

    test_path = Path('test_all')
    test_img_path = test_path / 'img'
    test_mask_path = test_path / 'mask'
    if not test_path.exists():
        test_path.mkdir()
        test_img_path.mkdir()
        test_mask_path.mkdir()

    for row in df_train.iterrows():
        pic_num = str(int(row[1][1]) + 1)
        img = Image.open(f'data/image/' + pic_num + '.png')
        img.save(f'train_all/img/' + pic_num + '.png')
        msk = Image.open(f'data/label/' + pic_num + '.png')
        msk.save(f'train_all/mask/' + pic_num + '.png')

    for row in df_test.iterrows():
        pic_num = str(int(row[1][1]) + 1)
        img = Image.open(f'data/image/' + pic_num + '.png')
        img.save(f'test_all/img/' + pic_num + '.png')
        msk = Image.open(f'data/label/' + pic_num + '.png')
        msk.save(f'test_all/mask/' + pic_num + '.png')


def overlap():
    Img = []
    Msk = []
    names = []
    for img in os.listdir("test/img"):
        img1 = Image.open("test/img/" + img)
        names.append(img)
        img1 = img1.copy()
        Img.append(img1)
    for msk in os.listdir("test/mask"):
        msk1 = Image.open("test/mask/" + msk)
        msk1 = msk1.copy()
        Msk.append(msk1)
    cont = 0
    label_path = Path("newimg")
    if not label_path.exists():
        label_path.mkdir()
    for img, msk, name in zip(Img, Msk, names):
        new_img = Image.blend(img, msk, 0.5)
        new_img.save("newimg/" + name)
        cont += 1


if __name__ == '__main__':
    # do_split()
    normal_train_test_split()
    overlap()
    data_set_k_fold()
