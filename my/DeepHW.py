# from torchvision import transforms
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# import torch.optim as optim
# import torch.nn.functional as F
# import torch.nn as nn
# import numpy as np
# import json
# import torch
# import os
#
#
# # 解析JSON文件
# def parseJSON(selectedAttrs, jsonPath):
#     """
#         selectedAttrs:选择的属性
#         jsonPath:json文件路径
#         return:
#             img_name_list:根据jsonPath返回train或者test下的所有素描图片名字列表  [sketch1_sketch0110.png,...]
#             attrs_list:每个素描图片对应的属性列表   [[0,0,1,..],[0,0,1,..],...]     [[样本1特征1,样本1特征2,...]，[样本2特征1,样本2特征2,...],...]
#     """
#     # 读文件
#     fp = open(jsonPath, 'r')
#     data = json.load(fp)
#     # 素描图片名字列表
#     img_name_list = list()
#     # 素描图片对应属性列表
#     attrs_list = list()
#     # 每个item是一张图片以及它的属性信息
#     for item in data:
#         # photo1/image0001 ==> sketch1_sketch0001.png
#         str = item['image_name'].replace('photo', 'sketch').replace("/", "_").replace('image', 'sketch')
#         str += '.png'
#         img_name_list.append(str)
#         itemAttrs = list()
#         for attr in selectedAttrs:
#             itemAttrs.append(item[attr])
#         attrs_list.append(itemAttrs)
#     return img_name_list, attrs_list
#
#
# # 训练集素描图像的目录
# trainDirOfImgs = '/home/lab401/zsy/data/FS2K/train/sketch'
# # 测试集素描图像的目录
# testDirOfImgs = '/home/lab401/zsy/data/FS2K/test/sketch'
#
#
# # 加载FS2K数据集的类Dataset
# class FS2K(Dataset):
#     def __init__(self, selectedAttrs, jsonPath, transform, mode="train"):
#         self.img_name_list, self.labels_list = parseJSON(selectedAttrs, jsonPath)
#         self.transform = transform
#         self.mode = mode
#
#     def __len__(self):
#         return len(self.img_name_list)
#
#     def __getitem__(self, index):
#         img_name = self.img_name_list[index]
#         img_labels = self.labels_list[index]
#         if (self.mode == "train"):
#             img_path = os.path.join(trainDirOfImgs, img_name)
#         if (self.mode == "test"):
#             img_path = os.path.join(testDirOfImgs, img_name)
#         image = Image.open(img_path).convert("RGB")
#         if self.transform != None:
#             image = self.transform(image)
#         return image, img_labels
#
#
# # Dataloader
# def get_loader(selectedAttrs, jsonPath, batch_size, transform=None, mode="train"):
#     dataset = FS2K(selectedAttrs, jsonPath, transform, mode)
#     data_loader = DataLoader(dataset=dataset,
#                              batch_size=batch_size,
#                              shuffle=(mode == 'train'),
#                              drop_last=True)
#     return data_loader
#
#
# class FeatureExtraction(nn.Module):
#     def __init__(self):
#         super(FeatureExtraction, self).__init__()
#         # The arguments for commonly used modules:
#         # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
#         # torch.nn.MaxPool2d(kernel_size, stride, padding)
#
#         # 卷积层
#         # input image size: [3, 128, 128]
#         # output image size: [256, 8, 8]
#         self.cnn_layers = nn.Sequential(
#             nn.Conv2d(3, 64, 3, 1, 1),
#             # 64 * 126 * 126
#             # 卷积层之后总会添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定。
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2, 0),
#             # 64 * 63 * 63
#
#             nn.Conv2d(64, 128, 3, 1, 1),
#             # 128 * 61 * 61
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2, 0),
#             # 128 * 31 * 31
#
#             nn.Conv2d(128, 256, 3, 1, 1),
#             # 256 * 29 * 29
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.MaxPool2d(4, 4, 0),
#             # 256 * 8 * 8
#         )
#
#     def forward(self, image):
#         return self.cnn_layers(image)
#
#
# class FullConnect(nn.Module):
#     def __init__(self, input_dim=256 * 8 * 8, output_dim=2):
#         super(FullConnect, self).__init__()
#
#         # 全连接层
#         # input image size: [256 * 8 * 8]
#         # output image size:[2]
#         self.fc_layers = nn.Sequential(
#             nn.Linear(input_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 32),
#             nn.ReLU(),
#             nn.Linear(32, 8),
#             nn.ReLU(),
#             nn.Linear(8, output_dim)
#         )
#         # self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         # 从1维展开
#         x = x.flatten(1)
#         return self.fc_layers(x)
#
#
# class MutiClassifier(nn.Module):
#     def __init__(self):
#         super(MutiClassifier, self).__init__()
#
#         self.featureExtractor = FeatureExtraction()
#
#         self.FC_hair = FullConnect()
#         # 更改
#         # self.FC_gender = FullConnect()
#         # self.FC_earring = FullConnect()
#         # self.FC_smile = FullConnect()
#         # self.FC_frontal = FullConnect()
#         # self.FC_style = FullConnect(output_dim=3)
#
#     def forward(self, image):
#         # 先经过卷积层提取特征  output image size:256 * 8 * 8
#         features = self.featureExtractor(image)
#         # 不同全连接层对不同属性进行分类
#         hair = self.FC_hair(features)
#         # 更改
#         # gender = self.FC_gender(features)
#         # earring = self.FC_earring(features)
#         # smile = self.FC_smile(features)
#         # frontal = self.FC_frontal(features)
#         # style = self.FC_style(features)
#
#         # hair:[0,1]  gender:[0,1]  earring:[0,1]  smile:[0,1]  frontal:[0,1]   style:[0,1,2]
#         # 更改
#         # return hair, gender, earring, smile, frontal, style
#         return hair
#
#
# config = {
#     # 训练集的json文件路径
#     "json_train_path": '/home/lab401/zsy/data/FS2K/anno_train.json',
#     # 测试集的json文件路径
#     "json_test_path": '/home/lab401/zsy/data/FS2K/anno_test.json',
#     # 选择的属性
#     # 更改
#     # "selectedAttrs":["hair","gender","earring","smile","frontal_face","style"],
#     "selectedAttrs": ["hair"],
#     # 模型保存路径
#     'save_path': '/home/lab401/zsy/code/DeepHW/model/model.pth',
#     # 超参
#     "Epoches": 50,
#     "batch_szie": 16,
#     "lr": 0.001
# }
#
# # 整理需要用到的变量
# Epoches = config["Epoches"]
# batch_size = config["batch_szie"]
# learning_rate = config["lr"]
# selectedAttrs = config["selectedAttrs"]
# json_train_path = config["json_train_path"]
# json_test_path = config["json_test_path"]
# device = "cuda" if torch.cuda.is_available() else "cpu"
# # device = "cpu"
# transform = transforms.Compose([
#     # Resize成正方形
#     transforms.Resize((128, 128)),
#     # 变为tensor变量
#     transforms.ToTensor(),
#     # 进行标准化(标准化就是要把图片3个通道中的数据整理到[-1, 1]区间)
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])
# train_loader = get_loader(selectedAttrs, json_train_path, batch_size, transform)
# test_loader = get_loader(selectedAttrs, json_test_path, batch_size, transform, "test")
# # 初始化模型，并将其放在指定的设备上
# model = MutiClassifier().to(device)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#
# # 记录训练每个epoch的平均loss
# train_loss_record = []
# # 记录验证每个epoch的平均acc
# valid_acc_record = []
# # 记录验证每个epoch的每个属性的acc
# valid_acc_item_record = {}
# # 验证集上最佳准确率
# best_acc = 0
#
# # 初始化
# for attr in selectedAttrs:
#     valid_acc_item_record[attr] = []
#
# for epoch in range(Epoches):
#     # ---------- Train Start ----------
#
#     # 设置模型为训练模式
#     model.train()
#
#     # 记录训练单个epoch的loss
#     batch_loss = 0
#
#     for batch_idx, data in enumerate(train_loader):
#         imgs, labels = data
#         # 将素描图片放入模型，得出预测值
#         # 更改
#         # hair, gender, earring, smile, frontal, style = model(imgs.to(device))
#         hair = model(imgs.to(device))
#         # 计算各个属性的交叉熵损失
#         # F.cross_entropy(input, target):input的维度为[batchsize,classes,width,height]，target的维度为[batchsize,width,height]。
#         lossOfHair = F.cross_entropy(hair, labels[0].to(device))
#         # 更改
#         # lossOfGender = F.cross_entropy(gender,labels[1].to(device))
#         # lossOfEarring = F.cross_entropy(earring,labels[2].to(device))
#         # lossOfSmile = F.cross_entropy(smile,labels[3].to(device))
#         # lossOfFrontal = F.cross_entropy(frontal,labels[4].to(device))
#         # lossOfStyle = F.cross_entropy(style,labels[5].to(device))
#         # 计算总损失
#         # 更改
#         # batch_total_loss = lossOfHair + lossOfGender + lossOfEarring + lossOfSmile + lossOfFrontal + lossOfStyle
#         batch_total_loss = lossOfHair
#         # 应首先清除上一步中存储在参数中的梯度
#         optimizer.zero_grad()
#         # 梯度反传
#         batch_total_loss.backward()
#         # 更新参数
#         optimizer.step()
#         # 累加这个batch的loss
#         batch_loss += batch_total_loss.item()
#
#         # 计算一个epoch的平均损失 epoch_loss
#     epoch_average_loss = batch_loss / (batch_idx + 1)
#     # 记录每个epoch的平均损失
#     train_loss_record.append(epoch_average_loss)
#     # 打印loss信息
#     print("Epoch: %d/%d, loss: %.4f" % (epoch, Epoches, epoch_average_loss))
#
#     # ---------- Train End ----------
#
#     # ---------- Valid Start ----------
#
#     # 调整为评估模式
#     model.eval()
#
#     # 统计单个epoch正确率
#     correct_dict = {}
#     # 保存单个epoch预测值
#     predict_dict = {}
#     # 保存单个epoch label
#     label_dict = {}
#
#     # 初始化
#     for attr in selectedAttrs:
#         correct_dict[attr] = 0
#
#     # 批量迭代验证集
#     for batch_idx, data in enumerate(test_loader):
#         imgs, labels = data
#         # 在验证集上不需要计算梯度
#         with torch.no_grad():
#             # 更改
#             # hair, gender, earring, smile, frontal, style = model(imgs.to(device))
#             hair = model(imgs.to(device))
#             # 用于存放模型预测batch_size个样本各个属性的数据
#             # 更改
#             # out_dict = {'hair': hair, 'gender': gender, 'earring': earring,
#             # 'smile': smile, 'frontal_face': frontal, 'style': style}
#             out_dict = {'hair': hair}
#             # 一个batch包含的样本数
#             # 更改
#             batch_len = len(out_dict['hair'])
#             # 计算准确率（比较batch中每个样本每个属性）
#             # i表示第几个样本
#             for i in range(batch_len):
#                 # 取出selectedAttrs中每一个选中的属性，以及其index
#                 for attr_idx, attr in enumerate(selectedAttrs):
#                     # out_dict[attr]：取出某个属性batch_size个预测值
#                     # out_dict[attr][i]：选择out_dict中attr属性的第i个样本的预测值（[0.2,0.8]）
#                     # np.argmax(out_dict[attr][i].data.cpu().numpy())：通过argmax获得下标0或1或2
#                     # 第i个样本attr属性值的预测值
#                     pred = np.argmax(out_dict[attr][i].data.cpu().numpy())
#                     # labels[attr_idx]：batch_size个样本的attr属性
#                     # labels[attr_idx].data.cpu().numpy()[i]：取出第i个样本的attr属性label值
#                     # 第i个样本每个属性值的label值
#                     true_label = labels[attr_idx].data.cpu().numpy()[i]
#                     # 判断第i个样本的预测值和label值是否相等
#                     if pred == true_label:
#                         # 如果相等则表示第i个样本的attr属性预测正确，该属性预测正确数加1
#                         correct_dict[attr] = correct_dict[attr] + 1
#
#     # 用于记录平均准确率 每个epoch后所有样本所有属性的总准确率 / 属性数
#     valid_average_acc = 0
#     # 计算每个epoch后每个属性的准确率(80 --> 80%)
#     for attr in selectedAttrs:
#         correct_dict[attr] = correct_dict[attr] * 100 / (len(test_loader) * batch_size)
#         # 记录验证每个epoch每个属性准确率
#         valid_acc_item_record[attr].append(correct_dict[attr])
#         valid_average_acc += correct_dict[attr]
#     valid_average_acc /= len(selectedAttrs)
#     # 记录验证每个epoch平均准确率
#     valid_acc_record.append(valid_average_acc)
#
#     # ---------- Valid End ----------
#
#     # ---------- Chase Best Model Start ----------
#
#     # 比较正确率并保存最佳模型
#     if valid_average_acc > best_acc:
#         best_acc = valid_average_acc
#         # 将最好的模型参数保存到指定路径
#         torch.save(model.state_dict(), config['save_path'])
#         print("Epoch: %d/%d, Best_acc: %.4f" % (epoch, Epoches, best_acc))
#
#         # ---------- Chase Best Model End ----------
