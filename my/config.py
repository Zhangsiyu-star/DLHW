"""
全局变量
"""
# 基本不变的参数
json_train_path = "/home/lab401/zsy/data/FS2K/anno_train.json"
json_test_path = "/home/lab401/zsy/data/FS2K/anno_test.json"
sketch_train_path = "/home/lab401/zsy/data/FS2K/train/sketch"
sketch_test_path = "/home/lab401/zsy/data/FS2K/test/sketch"

resnet18 = "/home/lab401/zsy/data/FS2K/resnet18.pth"
resnet34 = "/home/lab401/zsy/data/FS2K/resnet34.pth"
resnet50 = "/home/lab401/zsy/data/FS2K/resnet50.pth"
AlexNet = "/home/lab401/zsy/data/FS2K/alexnet.pth"
VGG16 = "/home/lab401/zsy/data/FS2K/vgg16.pth"

# 可能需要更改的参数
selectedAttrs = ['hair', 'gender', 'earring', 'smile', 'frontal_face', 'style']
# selectedAttrs = ['style']
DEVICE_ID = '0'  # 显卡ID
Epoches = 50
batch_size = 16
lr = 1e-5
# ("BaseModel", "Resnet18", "Resnet34", "Resnet50", "VGG16", "AlexNet")
modelName = 'AlexNet'
save_path = "/home/lab401/zsy/code/DeepHW/model/model.pth"
isPretrained = False