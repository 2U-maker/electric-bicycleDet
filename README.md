# **电梯电瓶车检测**

# 一、项目背景

	   电动车因为具有便捷、轻巧、无污染等优点，成为了广大民众出行的重要选择，但是电动车自身的电瓶存在安全隐患，目前仍然没有办法实现绝对的安全，近年来，电动车在室内、电梯内发生自燃、自爆的事件在新闻上常有报道，这些现象严重危害到了居民的生命、财产安全。
	   因此，本项目将以“电梯电瓶车检测”为首要目标进行开发，待项目开发成功以后，可以很容易地将本项目迁移到居民楼门口、办公楼门口、餐厅门口等众多关键场地，而且本项目的成本低廉，检测效率高，可应用的场景十分广泛，具有巨大的商业价值。
       
# 二、数据集简介

## 数据获取
* 1. 本项目的主要目的是“检测电瓶车”，因此需要大量的关于电瓶车的图片资料，并进行对象标注，标注完大量的电瓶车图片后进行模型训练、模拟。
为了增加检测的准确性，在抖音、西瓜视频等媒体平台上下载了不同种类的电瓶车的视频。
* 2. 将下载的视频放到Pr中进行剪辑，截取含有电动车的片段，再进行分帧，间隔取出照片，这样就获取了大量的不同种类的电瓶车照片。
* 3. 使用`LabelMe`进行图片的标注。
> [LabelMe使用方法](https://www.bilibili.com/video/BV1qq4y1X7uZ?p=2)

## 数据处理

* 1. 将标注完成的数据上传的`Ai Studio`上。
* 2. 数据集解压。


```python
!unzip -oq /home/aistudio/data1.zip
```

* 3. 安装`paddlex`。
> [paddlex使用文档](https://paddlex.readthedocs.io/zh_CN/release-1.3/index.html)


```python
!pip -q install paddlex
```

* 4.将解压后的数据转化成`voc`或者`coco`格式
> [转换方法](https://paddlex.readthedocs.io/zh_CN/release-1.3/data/annotation/object_detection.html)


```python
!paddlex --data_conversion --source labelme --to PascalVOC --pics data1/images --annotations data1/annotations --save_dir ./converted_dataset_dir
```

* 5.在转换后的文件夹中添加`labels.txt`， `labels.txt`中的每一行为一个标签。
> [数据集准备格式](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/docs/tutorials/PrepareDataSet.md#%E5%87%86%E5%A4%87%E8%AE%AD%E7%BB%83%E6%95%B0%E6%8D%AE)

## 将文件路径写入.txt


```python
import os
from tqdm import tqdm
import numpy

def path2txt(images_dir, anno_dir, txt_save_path, split_to_eval=False):
    data = []
    for line in os.listdir(images_dir):
        data.append([os.path.join(images_dir, line), os.path.join(anno_dir, line.replace('jpg', 'xml'))])
    
    if split_to_eval:
        numpy.random.shuffle(data)
        eval_data = data[:int(len(data)/10*2)]
        data = data[int(len(data)/10*2):]
        with open(txt_save_path.replace('train', 'val'), 'w') as f:
            for item in tqdm(eval_data):
                f.write('{} {}\n'.format(item[0], item[1]))

    with open(txt_save_path, 'w') as f:
        for item in tqdm(data):
            f.write('{} {}\n'.format(item[0], item[1]))


path2txt('converted_dataset_dir/JPEGImages', 'converted_dataset_dir/Annotations', 'converted_dataset_dir/train.txt', split_to_eval=True)
```

## 数据加载与预处理


```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from paddlex.det import transforms
import paddlex as pdx


# 定义训练和验证时的transforms
train_transforms = transforms.Compose([
    transforms.MixupImage(mixup_epoch=250),
    transforms.RandomDistort(),
    transforms.RandomExpand(),
    transforms.RandomCrop(),
    transforms.Resize(target_size=608, interp='RANDOM'),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(),
])

eval_transforms = transforms.Compose([
    transforms.Resize(target_size=608, interp='CUBIC'),
    transforms.Normalize(),
])

# 定义训练和验证所用的数据集
train_dataset = pdx.datasets.VOCDetection(
    data_dir='./',
    file_list='converted_dataset_dir/train.txt',
    label_list='converted_dataset_dir/labels.txt',
    transforms=train_transforms,
    shuffle=True)

eval_dataset = pdx.datasets.VOCDetection(
    data_dir='./',
    file_list='converted_dataset_dir/val.txt',
    label_list='converted_dataset_dir/labels.txt',
    transforms=eval_transforms)
```

# 三、模型选择与开发

## 模型选择

* 这里我们选择的模型是`YOLOv3-MobileNetV1`，其适合于后期边缘端的部署。
* 详细信息如下：

| **模型** | **模型大小** | **预测时间(ms/image）** | **BoxAP（%）**|
| -------- | -------- | -------- | -------- |
| [YOLOv3-MobileNetV1](https://paddlex.readthedocs.io/zh_CN/release-1.3/apis/models/detection.html) | 99.2MB | 11.834 | 9.3  |

## 模型训练


```python
# 初始化模型，并进行训练
num_classes = len(train_dataset.labels)

model = pdx.det.YOLOv3(num_classes=num_classes, backbone='MobileNetV1')


model.train(
    num_epochs=270,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    train_batch_size=32,
    learning_rate=0.000125,
    lr_decay_epochs=[213, 240],
    save_interval_epochs=90,
    save_dir='output/yolov3_mobilenetv1',
    metric='VOC',
    use_vdl=True)
```

# 四、效果展示

## 模型预测


```python
from paddlex.det import transforms
import paddlex as pdx


# 数据处理
eval_transforms = transforms.Compose([
    transforms.Resize(target_size=608, interp='CUBIC'),
    transforms.Normalize(),
])
# 模型载入
model = pdx.load_model('output/yolov3_mobilenetv1/epoch_150')
# 结果可视化
img_path = 'converted_dataset_dir/JPEGImages/074.jpg'
result = model.predict(img_file=img_path, transforms=eval_transforms)
pdx.det.visualize(image=img_path, result=result, threshold=0.03, save_dir='./')
```

## 预测结果可视化
<img style="display: block; margin: 0 auto;" src="https://ai-studio-static-online.cdn.bcebos.com/f9b3cbd5801a4c3b8c5c000ca912ccde78ba5f9367d44cac8541ddc895c0c928" width = "30%" height = "30%"/>

## 部署模型导出
> [部署模型导出](https://paddlex.readthedocs.io/zh_CN/release-1.3/deploy/export_model.html)


```python
!paddlex --export_inference --model_dir=output/yolov3_mobilenetv1/epoch_100 --save_dir=inference_model --fixed_input_shape=[608,608]
```

# 五、总结与升华
* 1. 由于未拿到硬件，部署部分等待更新~
* 2. 目前的数据较少，需要用`LabelMe`标注更多的数据；一个简单的办法是，用少量的数据去训练模型，在用模型去标注其他的数据，然后再训练。

# 个人简介

* 菜鸡一枚~，啥都不会，干饭第一！！！
* [我在AI Studio上获得钻石等级，点亮8个徽章，来互关呀~ https://aistudio.baidu.com/aistudio/personalcenter/thirdview/517701](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/517701)
