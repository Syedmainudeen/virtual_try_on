# virtual_try_on
virtual_try_on deep learning project using VITON dataset 
DATASET = https://www.kaggle.com/datasets/rkuo2000/viton-dataset/data
This project is on basic training on the GAN model for virtual try on using VITON dataset 
And make sure you provide correct directories and file paths. 
Personally I face issues in Data preprocessing where the dataset is showing None even the datas are present.


Initially we parsing the pose file which is in json format. First execute               
Loading the data for both training and testing subfolders and also load pose data for train and test labels
do Data preprocessing
and the checking is there a data availabe in color and label subfolders just checking
Then defining Generator and Discriminator
and then separate modules for logger, lr_scheduler visualization
In train_utils there is a training loop
in main_script.py is a main code for training the GAN where the all modules are accessed from this module


EXECUTION SEQUENCE
pose_parse.py
custom_dataset.py
color.py
label.py
preprocess.py
G_D.py
lr_scheduler.py
logger.py
visualization.py
train_utils.py
main_scrip.py

  This is the code to only train the model.
  If you want to evaluate your model and do testing your model  you have to do further coding 

  
