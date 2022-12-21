import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import csv
from statistics import mean, stdev


def plot_models(history, path, best_model_epochs):

    plt.plot(history["loss"], label='Training')
    plt.plot(history["val_loss"], label='Validation')

    # Add in a title and axes labels
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.axvline(x=best_model_epochs, color='g', linestyle=(0, (5, 1)),  label='best model')

    # Display the plot
    plt.legend(loc='best')
    plt.savefig(path + "loss_fig.png")

    plt.clf()

    plt.plot(history["dice_coef"], label='Training')
    plt.plot(history["val_dice_coef"], label='Validation')

    # Add in a title and axes labels
    plt.title('Training and Validation DICE')
    plt.xlabel('Epochs')
    plt.ylabel('Dice')
    plt.axvline(x=best_model_epochs, color='g', linestyle=(0, (5, 1)), label='best model')

    # Display the plot
    plt.legend(loc='best')
    plt.savefig(path + "dice_fig.png")

    plt.clf()

    plt.plot(history["iou"], label='Training')
    plt.plot(history["val_iou"], label='Validation')

    # Add in a title and axes labels
    plt.title('Training and Validation IOU')
    plt.xlabel('Epochs')
    plt.ylabel('IOU')
    plt.axvline(x=best_model_epochs, color='g', linestyle=(0, (5, 1)), label='best model')

    # Display the plot
    plt.legend(loc='best')
    plt.savefig(path + "iou_fig.png")

    plt.clf()


def joint_plot_models(history1, history2, path, best_model_epochs_1, best_model_epochs_2):

    plt.plot(history1["val_dice_coef"], label='UNet')
    plt.plot(history2["val_dice_coef"], label='D-UNet')

    # Add in a title and axes labels
    plt.title('Validation DICE')
    plt.xlabel('Epochs')
    plt.ylabel('Dice')
    plt.axvline(x=best_model_epochs_1, color='g', linestyle=(0, (5, 1)), label='best unet model')
    plt.axvline(x=best_model_epochs_2, color='purple', linestyle=(0, (5, 1)), label='best d_unet model')

    # Display the plot
    plt.legend(loc='best')
    plt.savefig(path + "dice_fig_vs.png")

    plt.clf()

    plt.plot(history1["val_iou"], label='UNet')
    plt.plot(history2["val_iou"], label='D-UNet')

    # Add in a title and axes labels
    plt.title('Validation IOU')
    plt.xlabel('Epochs')
    plt.ylabel('IOU')
    plt.axvline(x=best_model_epochs_1, color='g', linestyle=(0, (5, 1)), label='best unet model')
    plt.axvline(x=best_model_epochs_2, color='purple', linestyle=(0, (5, 1)), label='best d_unet model')

    # Display the plot
    plt.legend(loc='best')
    plt.savefig(path + "iou_fig_vs.png")

    plt.clf()


def compress(path, new_path, only_check_path):
    DF = []
    for cv in os.listdir(path):
        DF.append(pd.read_csv(path + cv))
    cont = 1
    f = open(new_path, 'w')
    f1 = open(only_check_path, 'w')
    epochs = 1000
    columns = ["Epochs", "loss", "std_loss", "iou", "std_iou", "dice_coef", "std_dice",
               "val_loss", "std_val_loss", "val_iou", "std_val_iou", "val_dice_coef", "std_val_dice"]
    writer = csv.writer(f)
    writer.writerow(columns)
    writer1 = csv.writer(f1)
    writer1.writerow(columns)
    for i in range(epochs):
        I = []
        D = []
        L = []
        vI = []
        vD = []
        vL = []
        for d in DF:
            L.append(d.iloc[i]['loss'])
            I.append(d.iloc[i]['iou'])
            D.append(d.iloc[i]['dice_coef'])
            vL.append(d.iloc[i]['val_loss'])
            vI.append(d.iloc[i]['val_iou'])
            vD.append(d.iloc[i]['val_dice_coef'])
        avgL = mean(L)
        avgI = mean(I)
        avgD = mean(D)
        avgvL = mean(vL)
        avgvI = mean(vI)
        avgvD = mean(vD)

        sL = stdev(L)
        sI = stdev(I)
        sD = stdev(D)
        svL = stdev(vL)
        svI = stdev(vI)
        svD = stdev(vD)

        r = [str(cont), str(avgL), str(sL), str(avgI), str(sI), str(avgD), str(sD),
             str(avgvL), str(svL), str(avgvI), str(svI), str(avgvD), str(svD)]
        writer.writerow(r)
        if cont % 100 == 0:
            writer1.writerow(r)
        cont += 1
    f.close()
    f1.close()


if __name__ == "__main__":

    results_path = Path('results')
    unet_32_path = results_path / 'unet_32_visual'
    unet_normal_path = results_path / 'unet_normal_visual'
    unet_32_path.mkdir()
    unet_normal_path.mkdir()

    path = 'results/unet_normal/'
    visual_path = 'results/unet_normal_visual/'
    new_path_1 = 'results/unet_normal_visual/mean_history.csv'
    only_check_path = 'results/unet_normal_visual/mean_history_checkpoints.csv'

    compress(path, new_path_1, only_check_path)
    plot_models(pd.read_csv(new_path_1), visual_path, best_model_epochs=600)

    path = 'results/unet_32/'
    visual_path = 'results/unet_32_visual/'
    new_path_2 = 'results/unet_32_visual/mean_history.csv'
    new_write = 'results/unet_32_visual/table.csv'
    only_check_path = 'results/unet_32_visual/mean_history_checkpoints.csv'

    compress(path, new_path_2, only_check_path)
    plot_models(pd.read_csv(new_path_2), visual_path, best_model_epochs=800)

    joint_plot_models(pd.read_csv(new_path_1), pd.read_csv(new_path_2), path='results/',
                      best_model_epochs_1=600, best_model_epochs_2=800)


