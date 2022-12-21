# Unet_ICH_Detector
Brain stroke is the second-leading cause of death worldwide after heart disease, and one of the most concerning types are the intracranial hemorrhages. This type of bleeding, caused by ruptures of blood vessels within the brain, affects the brain, prevents cell oxygenation, and causes damage to the nerves. Although recent advances in medicine have been the salvation for many patients, doctors are still subject to human errors when detecting and segmenting intracranial hemorrhages due to long working hours. For this reason, deep-learning models have been introduced to help reduce errors in these medical field. In this regard, we proposed a new deep-learning method called D-Unet based on the standard U-net architecture to successfully detect and segment intracranial hemorrhage lesions on a data set of computerized tomography scan images belonging to 82 patients. Both the D-Unet and U-net were trained under the same experimental conditions using a stratified ten-fold cross-validation schema, and the obtained means of IoU and DICE coefficient scores of 0.72 (0.05 of standard deviation) and 0.84 (0.03 of standard deviation) for the D-Unet, and 0.65 (0.04 of standard deviation) and 0.79 (0.03 of standard deviation) for the U-net, demonstrated that D-Unet tends to perform better than its baseline method. Also, the best-selected D-Unet model in the training stage was validated in an external test set, reaching scores of 0.86 for IoU and 0.89 for DICE. This performance evaluation on the test data set confirmed the model’s quality and generalization capacity, making it successful in detecting and segmenting ICH of different types, shapes, sizes, and locations.

## How to use it
This code can work with different types of data sets. If these is the case, skip the first two steps.
### Step 1: 
Download the original data set from Physionet. In this case, we used an upgraded version which was found on: https://physionet.org/content/ct-ich/1.3.1/. You will not need split_raw_data.py from that file since an upgraded version of that script is included in this project
### Step 2:
Run file create_data_set.py.
### Step 3:
Run main_kfold.py if you want to do a stratified k-fold cross validation. Else run main.py.
You can change the model you want to train between UNet and UNet1 (our modified model). They are both defined in models.py.
### Step 4:
If you used the k-fold technique, run trials.py to obtain a table with the mean scores and the graphics.
### Step 5:
Run tests.py to obtain results on the test set and get the predicted masks.
