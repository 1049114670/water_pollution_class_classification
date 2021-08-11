#!/usr/bin/env python
# coding: utf-8

# ### 用于第二届华录杯智慧水利河道污染等级分类
# 

# ### 1、解压数据集
# 1.数据集格式：train中有图片和对应标注好标签和污染等级的csv文件；；test文件只有图片。  
# 2.train的数据集8:2划分为训练集和验证集  

# In[1]:


get_ipython().system('pwd   #显示目录')
get_ipython().system('unzip -oq data/data101229/data.zip')


# ### 2、导入库

# In[2]:


# 导入所需要的库
from sklearn.utils import shuffle
import os
import pandas as pd
import numpy as np
from PIL import Image

import paddle
import paddle.nn as nn
from paddle.io import Dataset
import paddle.vision.transforms as T  # 数据变换增强模块
import paddle.nn.functional as F
from paddle.metric import Accuracy  # 准确率模块

import warnings
warnings.filterwarnings("ignore")


# ### 3、读取数据

# In[4]:


# 读取数据
train_images = pd.read_csv('data/train_data/train_label.csv', usecols=['image_name','label'])  # 读取文件名和类别
# print(train_images)

# labelshuffling，定义标签打乱模块
def labelShuffling(dataFrame, groupByName='label'):

    groupDataFrame = dataFrame.groupby(by=[groupByName])
    labels = groupDataFrame.size()
    print("length of label is ", len(labels))
    maxNum = max(labels)
    lst = pd.DataFrame()
    for i in range(len(labels)):
        print("Processing label  :", i)
        tmpGroupBy = groupDataFrame.get_group(i)
        createdShuffleLabels = np.random.permutation(np.array(range(maxNum))) % labels[i]  # 随机排列组合
        print("Num of the label is : ", labels[i])
        lst=lst.append(tmpGroupBy.iloc[createdShuffleLabels], ignore_index=True)
        # print("Done")
    # lst.to_csv('test1.csv', index=False)
    return lst

# 由于数据集是按类规则排放的，应该先打乱再划分

df = labelShuffling(train_images)
df = shuffle(df)
# 划分训练集和验证集 8:2;;后面好像测试集直接引用了验证集的图
all_size = len(train_images)
print("训练集大小：", all_size)
train_size = int(all_size * 0.8)
train_image_list = df[:train_size]
val_image_list = df[train_size:]

print("shuffle后数据集大小：", len(df))

train_image_path_list = df['image_name'].values
label_list = df['label'].values
label_list = paddle.to_tensor(label_list, dtype='int64')# 该API通过已知的 data 来创建一个tensor(张量)
train_label_list = paddle.nn.functional.one_hot(label_list, num_classes=4)# 将张量转换为one hot形式的张量，同一时间只有一个点被激活=1，其余为0

val_image_path_list = val_image_list['image_name'].values
val_label_list = val_image_list['label'].values
val_label_list = paddle.to_tensor(val_label_list, dtype='int64')
val_label_list = paddle.nn.functional.one_hot(val_label_list, num_classes=4)


# In[7]:


# 定义数据预处理,数据增广方法
data_transforms = T.Compose([
    T.RandomResizedCrop(224, scale=(0.8, 1.2), ratio=(3. / 4, 4. / 3), interpolation='bilinear'),
    T.RandomHorizontalFlip(),# 随机水平翻转
    T.RandomVerticalFlip(),# 随机垂直翻转
    T.RandomRotation(15),#随机旋转一定角度，根据参数，生成指定的旋转范围-15°,15°
    T.Transpose(),    # HWC -> CHW
    T.Normalize(    # 归一化
        mean=[127.5, 127.5, 127.5],       
        std=[127.5, 127.5, 127.5],
        to_rgb=True)    
])


# In[8]:


# 构建Dataset
#需要用GPU跑，
import paddle
class MyDataset(paddle.io.Dataset):
    """
    步骤一：继承paddle.io.Dataset类
    """
    def __init__(self, train_img_list, val_img_list, train_label_list, val_label_list, mode='train'):
        """
        步骤二：实现构造函数，定义数据读取方式，划分训练和测试数据集;;
        ### 应该是验证集吧；直接将训练集数据分成了训练和验证，还得想想怎么测试测试集的图片
        """
        super(MyDataset, self).__init__()
        self.img = []
        self.label = []
        # 借助pandas读csv的库
        self.train_images = train_img_list
        self.val_images = val_img_list  ###############
        self.train_label = train_label_list
        self.val_label = val_label_list  ##############
        if mode == 'train':
            # 读train_images的数据
            for img,la in zip(self.train_images, self.train_label):
                self.img.append('data/train_data/train_image/' + img)
                self.label.append(la)
        else:
            # 读test_images的数据
            for img,la in zip(self.val_images, self.val_label):   #############
                self.img.append('data/train_data/train_image/' + img)
                self.label.append(la)

         

    def load_img(self, image_path):
        # 实际使用时使用Pillow相关库进行图片读取即可，这里我们对数据先做个模拟
        image = Image.open(image_path).convert('RGB')
        return image

    def __getitem__(self, index):
        """
        步骤三：实现__getitem__方法，定义指定index时如何获取数据，并返回单条数据（训练数据，对应的标签）
        """
        image = self.load_img(self.img[index])
        label = self.label[index]
        # label = paddle.to_tensor(label)
        
        return data_transforms(image), paddle.nn.functional.label_smooth(label)

    def __len__(self):
        """
        步骤四：实现__len__方法，返回数据集总数目
        """
        return len(self.img)

BATCH_SIZE = 128  # 可调，修改batch_size大小,一般为2的指数，太大容易内存溢出，可以通过看内存占用情况进行调节，到90%-95%
PLACE = paddle.CUDAPlace(0)

# train_loader
train_dataset = MyDataset(
    train_img_list=train_image_path_list, 
    val_img_list=val_image_path_list, 
    train_label_list=train_label_list, 
    val_label_list=val_label_list, 
    mode='train')
train_loader = paddle.io.DataLoader(
    train_dataset, # 数据加载地址
    places=PLACE, #数据需要放置到的Place列表,在动态图模式中，此参数列表长度必须是1。默认值为None。
    batch_size=BATCH_SIZE, #  每mini-batch中样本个数,
    shuffle=True, # 生成mini-batch索引列表时是否对索引打乱顺序
    num_workers=0)

# val_loader
val_dataset = MyDataset(
    train_img_list=train_image_path_list, 
    val_img_list=val_image_path_list, 
    train_label_list=train_label_list, 
    val_label_list=val_label_list, 
    mode='test')
val_loader = paddle.io.DataLoader(
    val_dataset, 
    places=PLACE, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=0)


# ### 4、定义模型

# In[9]:


# 定义模型
class MyNet(paddle.nn.Sequential):
    def __init__(self):
        super(MyNet, self).__init__(
                paddle.vision.models.mobilenet_v2(pretrained=True, scale=1.0),
                paddle.nn.Linear(1000, 4)# 线性变换层输入单元和输出单元的数目，这里是四分类
        )

model = MyNet()
model = paddle.Model(model)
model.summary((1, 3, 224, 224))


# ### 5、模型训练 Trick

# In[10]:


def make_optimizer(parameters=None, momentum=0.9, weight_decay=5e-4, boundaries=None, values=None):
    
    lr_scheduler = paddle.optimizer.lr.PiecewiseDecay(
        boundaries=boundaries, 
        values=values,
        verbose=False)

    lr_scheduler = paddle.optimizer.lr.LinearWarmup(
        learning_rate=base_lr,
        warmup_steps=20,
        start_lr=base_lr / 5.,
        end_lr=base_lr,
        verbose=False)

    # optimizer = paddle.optimizer.Momentum(
    #     learning_rate=lr_scheduler,
    #     weight_decay=weight_decay,
    #     momentum=momentum,
    #     parameters=parameters)

    optimizer = paddle.optimizer.Adam(
        learning_rate=lr_scheduler,
        weight_decay=weight_decay,
        parameters=parameters)

    return optimizer


base_lr = 0.001
boundaries = [33, 44]

optimizer = make_optimizer(boundaries=boundaries, values=[base_lr, base_lr*0.1, base_lr*0.01], parameters=model.parameters())
# learning_rate = base_lr       if epoch < 33
# learning_rate = base_lr*0.1   if 33 <= epoch < 44
# learning_rate = base_lr*0.01  if 44 <= epoch
# 配置模型所需的部件，比如优化器、损失函数和评价指标
model.prepare(
    optimizer=optimizer,
    loss=paddle.nn.CrossEntropyLoss(soft_label=True),
    metrics=paddle.metric.Accuracy(topk=(1, 5))
)

# callbacks
visualdl = paddle.callbacks.VisualDL('./visualdl/MobileNetV2')
earlystop = paddle.callbacks.EarlyStopping( # acc不在上升时停止
    'acc',
    mode='max',
    patience=2,
    verbose=1,
    min_delta=0,
    baseline=None,
    save_best_model=True)

model.fit(
    train_loader,
    val_loader,
    epochs=50,
    save_freq=5,
    save_dir='checkpoint/MobileNetV2',
    callbacks=[visualdl, earlystop],
    verbose=1
)


# In[11]:


# 训练模型保存
model.save('work/model/best_model')  # save for training


# ### 6、模型评估

# In[12]:


model.load('work/model/best_model')
model.prepare(
    loss=paddle.nn.CrossEntropyLoss(soft_label=True),
    metrics=paddle.metric.Accuracy()
)
result = model.evaluate(val_loader, batch_size=1, verbose=1)
print(result)


# In[ ]:


model.save('work/model/best_model', training=False)  # save for inference


# ### 7、总结
# 主要是参加了paddlepaddle的培训课程，借项目来练练手，熟悉一下流程
# 里面需要用到的[API](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html)，可以自己查阅使用
