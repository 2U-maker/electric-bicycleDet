import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from paddlex.det import transforms
import paddlex as pdx
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
+
# 模型载入
model = pdx.load_model('output/yolov3_mobilenetv1/epoch_150')
# 结果可视化
img_path = 'converted_dataset_dir/JPEGImages/074.jpg'
result = model.predict(img_file=img_path, transforms=eval_transforms)
pdx.det.visualize(image=img_path, result=result, threshold=0.03, save_dir='./')
