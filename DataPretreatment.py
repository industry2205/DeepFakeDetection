#!/usr/bin/env python
# coding: utf-8

# In[13]:


import os
from PIL import Image
import cv2, sys, re
import pandas as pd
import numpy as np
import random
import numpy as np
import glob
from sklearn.metrics import log_loss
import tensorflow as tf

# Keras
# import keras.backend as K
# from keras.models import load_model
# from keras import models,Sequential,layers
# from keras.models import Model, Input
# from keras.layers import Conv2D, SeparableConv2D, Dense, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D, DepthwiseConv2D
# from keras.layers import Activation, BatchNormalization, Dropout, Flatten, Reshape, ReLU, Dense, multiply, Softmax, Flatten, Input
# from keras.layers import Add, Input
# from keras.utils import to_categorical
# from keras.callbacks import Callback
# from keras.optimizers import SGD,Adam
# from keras.regularizers import l2
# from keras.datasets import cifar10

# from sklearn.ensemble import RandomForestClassifier


# In[ ]:


# 전처리 데이터를 불러오기 위한 Directory 지정
all_train_dirs = glob.glob('/kaggle/input/all-data/dfdc_train_part/' + 'dfdc_train_part_*')


# In[ ]:


# 이미지를 CascadeClassifier 함수 활용하여 눈으로 인식하는 갯수를 이용하여
# 1개일 시 옆모습, 1개 이외일 시 정면으로 지정하여 데이터셋을 나눔
eye_cascade = cv2.CascadeClassifier('/kaggle/input/haar-cascades-for-face-detection/haarcascade_eye.xml')

def detection_eyes(a):
    roi_gray = a[0:160,0:160]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    cnt_eyes = 0
    for (ex,ey,ew,eh) in eyes:
        cnt_eyes += 1
    return cnt_eyes


# 위의 함수를 이용하여 이미지를 불러와 Numpy 형식으로 변환 (옆,앞모습 데이터셋도 나눔)
def read_image(path,file_list,label_img):
    # a는 한 동영상에서 생성된 이미지의 번호를 입력한다. (원하는 이미지를 저장하게 됨)
    a = [150,250]
    im = []
    label = []
    for i in file_list:
        path_dir = path +'/' + i
        for j in a:
            try:
                # 이미지를 Numpy 형식으로 변환
                img = Image.open(path_dir + '/' + str(j) +'.png')
                arr_img = np.array(img)
                # 눈의 갯수로 옆, 앞모습을 나눔
                if detection_eyes(arr_img) == 1:
#                 if detection_eyes(arr_img) != 1:
                    im.append(list(arr_img))
                    label_list = np.array(label_img.iloc[:, [2]])
                # Labeling 작업
                    if i + '.mp4' in label_list:
                        label.append(1)  # Deepfake
                    else:
                        label.append(0)
            except:
                pass
    return im, label


# In[ ]:


# read_image 함수를 이용하여 지정한 이미지를 저장하는 작업
X = []
y = []
for i in range(len(all_train_dirs)):
    path = all_train_dirs[i]
    file_list = os.listdir(path)
    label = pd.read_csv(all_train_dirs[i] + '/metadata.csv',delimiter=',')
    img,label = read_image(path,file_list,label)
    X += img
    y += label

X = np.array(X)
y = np.array(y).reshape(-1,1)


# In[ ]:


# savez를 이용하여 이미지와 라벨을 한번에 저장
# (데이터를 npz형태로 저장하여 사용함으로써, 데이터 불러오는 시간을 줄임)
np.savez('Facial.npz',X,y)
# np.savez('Side.npz',X,y)


# In[15]:


# Keras Version

# Numpy 데이터 불러오기 (이미지, 라벨이 묶여서 저장되어 있음)
# arr_0은 이미지, arr_1은 label
# 1만개의 Deepfake, 5만개의 Real 이미지가 순서대로 있음
# np.r_을 사용하여 Numpy로 불러온 이미지를 필요한 만큼 쓰기 위해 사용
# (224,224,3) Shape을 (-1,224,224,3) Shape으로 붙임

arr = np.load('/kaggle/input/newdata/new_data_side.npz','r')

X = arr['arr_0']
y = arr['arr_1']


# 데이터 Shuffle

s = np.arange(X.shape[0])
np.random.shuffle(s)

X = X[s]
y = y[s]


# 데이터 Augumentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# flow_from_directory는 폴더를 지정해주지만, flow는 데이터를 직접 입력시킨다
# flow에 데이터를 입력할 때, (이미지, 라벨) 형식으로 사용
train_generator = train_datagen.flow(
    # This is the target directory
    (X_train,y_train),
    batch_size=128,
)

validation_generator = test_datagen.flow(
    (X_val,y_val),
    batch_size=128,
)


# In[ ]:


# Pytorch Version

arr = np.load('/kaggle/input/newdata/new_data_side.npz','r')

X = arr['arr_0']
y = arr['arr_1']

# 데이터 Augumentation을 위해 Validation Set의 비율을 크게 함
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.35, random_state=1)
del X
del y

# img_size는 EfficientNet을 사용하기 때문에, 이미지를 많이 사용할 수 있도록
# 120X120의 형태로 진행. (160, 224 크기로 확인하였으나, 120 사이즈의 결과가 좋았음)
# mean, std는 이미지 RGB 각각을 따로 지정하여 정규화
img_size = 120
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

# 데이터 Augumentation
class ImageTransform_train:
    def __init__(self, size, mean, std):
        self.data_transform = transforms.Compose([
                transforms.Resize((size, size), interpolation=Image.BILINEAR),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

    def __call__(self, img):
        return self.data_transform(img)
    
class ImageTransform_val:
    def __init__(self, size, mean, std):
        self.data_transform = transforms.Compose([
                transforms.Resize((size, size), interpolation=Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

    def __call__(self, img):
        return self.data_transform(img)
    
transform_train = ImageTransform_train(img_size, mean, std)
transform_val = ImageTransform_val(img_size, mean, std)


# 위의 함수를 이용하여 Numpy형식을 Tensor로 변환 및 Augumentation
# 메모리 활용을 위해 필요없는 데이터는 즉시 삭제
img_list = []

for image in X_train:     
    try:
        image = Image.fromarray(image)
        image = transform_train(image)
        img_list.append(image)
    except:
        img_list.append(None)

del X_train

img_list_val = []

for image in X_val:          
    try:
        image = Image.fromarray(image)
        image = transform_val(image)
        img_list_val.append(image)
    except:
        img_list_val.append(None)
    
del X_val

# Label Numpy형식 데이터를 Tensor로 변환
y_train = torch.from_numpy(y_train).float()
y_val = torch.from_numpy(y_val).float()


# 훈련 시 Loss 값을 시각화 하기 위해 사용
train_len = len(X_train)
val_len = len(X_val)
dataset_sizes = {'train':train_len,'val':val_len}


# 데이터 Loader
# Train Set과 Validation Set을 나눠서 진행
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


class JoinDataset_train(Dataset):
    def __init__(self):
        self.len = y_train.shape[0]
        self.x_data = img_list
        self.y_data = y_train

    """ Diabetes dataset."""
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

trainset = JoinDataset_train()
train_loader = DataLoader(dataset=trainset,batch_size=64,shuffle=True)


class JoinDataset_val(Dataset):
    def __init__(self):
        self.len = y_val.shape[0]
        self.x_data = img_list_val
        self.y_data = y_val

    """ Diabetes dataset."""
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
    
valset = JoinDataset_val()
val_loader = DataLoader(dataset=valset,batch_size=64,shuffle=True)

dataloders = {'train':train_loader,'val':val_loader}


# In[ ]:


위는 데이터 정제
-----------------------------------------절취선
아래는 Features 확인


# In[ ]:


# Feature map 시각화 (원하는 Layer를 지정하여 Features를 확인할 수 있다)
# PPT 이미지에 포함시킴

# 이미지 불러오기
def load_image(img_path, target_size):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
    img_tensor = tf.keras.preprocessing.image.img_to_array(img)
    # 배치 사이즈 추가 + 스케일링 결과 반환
    return img_tensor[np.newaxis] / 255     # (1, 150, 150, 3)


# 첫 번째 등장하는 컨볼루션 레이어의 모든 피처맵(32개) 출력
def show_first_feature_map(loaded_model, img_path):
    # layers[?] : Layer 번호를 지정하여 특정 Layer에서 생성되는 Feature를 확인할 수 있다
    first_output = loaded_model.layers[2].output
    print(first_output.shape, first_output.dtype)

    # 1개의 출력을 갖는 새로운 모델 생성
    model = tf.keras.models.Model(inputs=loaded_model.input, outputs=first_output)

    # 입력으로부터 높이와 너비를 사용해서 target_size에 해당하는 튜플 생성
    target_size = (loaded_model.input.shape[2], loaded_model.input.shape[2])
    img_tensor = load_image(img_path, target_size)

    print(loaded_model.input.shape)     
    print(img_tensor.shape)          

    first_activation = model.predict(img_tensor)

    print(first_activation.shape)       
    print(first_activation[0, 0, 0]) 

    # first_activation 콜론(:)은 높이와 너비를 가리키는 차원의 모든 데이터
    
    plt.figure(figsize=(16, 8))
    for i in range(first_activation.shape[-1]):
        plt.subplot(4, 8, i + 1)

        # 눈금 제거. fignum은 같은 피켜에 연속 출력
        plt.axis('off')
        plt.matshow(first_activation[0, :, :, i], cmap='gray', fignum=0)
    plt.tight_layout()
    plt.show()

# 위의 함수를 이용하여, 지정 이미지에서 생성되는 Feature를 출력
show_first_feature_map(model,'/kaggle/input/deepfake-detection-faces-part-3-0/vezsoxophr/150.png')


# In[3]:


get_ipython().system('pip install ipynb-py-convert')


# In[6]:


get_ipython().system('ipynb-py-convert DataPretreatment.ipynb DataPretreatment.py')


# In[11]:


export LANG=C.UTF-8
get_ipython().system('export PYTHONIOENCODING=utf-8')
get_ipython().system('export PYTHONUTF8=1')


# In[12]:


# %%

