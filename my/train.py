from torch import optim
from torchvision import transforms
from DataLoader import get_loader
from model import MultiAttrsClassifier
import torch.nn.functional as F
import numpy as np
import torch
import config


def train():
    # 整理需要用到的变量
    Epoches = config.Epoches
    batch_size = config.batch_size
    learning_rate = config.lr
    modelName = config.modelName
    isPretrained = config.isPretrained
    selectedAttrs = config.selectedAttrs
    device = torch.device("cuda:" + str(config.DEVICE_ID) if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        # Resize成正方形
        transforms.Resize((128, 128)),
        # 变为ten
        transforms.ToTensor(),
        # 进行标准化(标准化就是要把图片3个通道中的数据整理到[-1, 1]区间)
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    train_loader = get_loader(batch_size, selectedAttrs, "train", transform)
    test_loader = get_loader(batch_size, selectedAttrs, "test", transform)
    model = MultiAttrsClassifier(modelName, isPretrained).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # 记录训练集每个step的损失
    train_step_loss_record = []
    # 记录验证集每个step的损失
    valid_step_loss_record = []
    # 验证集上最佳准确率
    best_acc = 0
    # 记录验证集每个属性的acc
    valid_acc_attr_record = {}
    # 记录验证集最好情况是每个属性的acc
    best_valid_acc_attr = {}
    # 记录验证集除了style的每个属性的混淆矩阵
    valid_confusion_matrix_record = {}

    # 初始化
    for attr in selectedAttrs:
        valid_acc_attr_record[attr] = []

    # 开始训练
    for epoch in range(Epoches):
        # ---------- Train Start ----------

        # 设置模型为训练模式
        model.train()

        for batch_idx, data in enumerate(train_loader):
            imgs, labels = data
            # 将素描图片放入模型，得出预测值
            # 修改
            hair, gender, earring, smile, frontal, style = model(imgs.to(device))
            # style = model(imgs.to(device))
            # 计算各个属性的交叉熵损失
            # F.cross_entropy(input, target):input的维度为[batchsize,classes,width,height]，target的维度为[batchsize,width,height]。
            # 修改
            # lossOfStyle = F.cross_entropy(style, labels[0].to(device))
            lossOfHair = F.cross_entropy(hair, labels[0].to(device))
            lossOfGender = F.cross_entropy(gender, labels[1].to(device))
            lossOfEarring = F.cross_entropy(earring, labels[2].to(device))
            lossOfSmile = F.cross_entropy(smile, labels[3].to(device))
            lossOfFrontal = F.cross_entropy(frontal, labels[4].to(device))
            lossOfStyle = F.cross_entropy(style, labels[5].to(device))
            # 计算总损失
            # 修改
            batch_train_total_loss = lossOfHair + lossOfGender + lossOfEarring + lossOfSmile + lossOfFrontal + lossOfStyle
            # batch_train_total_loss = lossOfStyle
            # 记录每个step的损失
            train_step_loss_record.append(batch_train_total_loss.item())
            # 应首先清除上一步中存储在参数中的梯度
            optimizer.zero_grad()
            # 梯度反传
            batch_train_total_loss.backward()
            # 更新参数
            optimizer.step()
            # 打印信息
            if batch_idx % 20 == 0:
                print("Epoch: %d/%d, training batch_idx:%d , loss: %.4f" % (epoch, Epoches, batch_idx, batch_train_total_loss.item()))

        # ---------- Train End ----------

        # ---------- Valid Start ----------

        # 调整为评估模式
        model.eval()

        # 统计单个epoch正确率
        correct_dict = {}
        
        # 初始化
        for attr in selectedAttrs:
            correct_dict[attr] = 0
        
        # 记录单个epoch验证集除了style的每个属性的混淆矩阵
        valid_confusion_matrix_record_temp = {}
        
        # 初始化
        for attr in selectedAttrs:
            if attr != "style":
                valid_confusion_matrix_record_temp[attr] = {}
                # TP : 0 --> 0
                valid_confusion_matrix_record_temp[attr]["TP"] = 0
                # TN : 0 --> 1
                valid_confusion_matrix_record_temp[attr]["TN"] = 0
                # FT : 1 --> 0
                valid_confusion_matrix_record_temp[attr]["FP"] = 0
                # FN : 1 --> 1
                valid_confusion_matrix_record_temp[attr]["FN"] = 0        

        # 批量迭代验证集
        for batch_idx, data in enumerate(test_loader):
            imgs, labels = data
            # 在验证集上不需要计算梯度
            with torch.no_grad():
                # 修改
                hair, gender, earring, smile, frontal, style = model(imgs.to(device))
                # style = model(imgs.to(device))
                # 计算各个属性的交叉熵损失
                # 修改
                # lossOfStyle = F.cross_entropy(style, labels[0].to(device))
                lossOfHair = F.cross_entropy(hair, labels[0].to(device))
                lossOfGender = F.cross_entropy(gender, labels[1].to(device))
                lossOfEarring = F.cross_entropy(earring, labels[2].to(device))
                lossOfSmile = F.cross_entropy(smile, labels[3].to(device))
                lossOfFrontal = F.cross_entropy(frontal, labels[4].to(device))
                lossOfStyle = F.cross_entropy(style, labels[5].to(device))
                # 计算总损失
                # 修改
                batch_valid_total_loss = lossOfHair + lossOfGender + lossOfEarring + lossOfSmile + lossOfFrontal + lossOfStyle
                # batch_valid_total_loss = lossOfStyle
                # 记录每个step的损失
                valid_step_loss_record.append(batch_valid_total_loss.item())
                # 用于存放模型预测batch_size个样本各个属性的数据
                # 修改
                out_dict = {'hair': hair, 'gender': gender, 'earring': earring, 'smile': smile, 'frontal_face': frontal,
                            'style': style}
                # out_dict = {'style': style}
                # 一个batch包含的样本数
                # 修改
                batch_len = len(out_dict['hair'])
                # 计算准确率（比较batch中每个样本每个属性）
                # i表示第几个样本
                for i in range(batch_len):
                    # 取出selectedAttrs中每一个选中的属性，以及其index
                    for attr_idx, attr in enumerate(selectedAttrs):
                        # out_dict[attr]：取出某个属性batch_size个预测值
                        # out_dict[attr][i]：选择out_dict中attr属性的第i个样本的预测值（[0.2,0.8]）
                        # np.argmax(out_dict[attr][i].data.cpu().numpy())：通过argmax获得下标0或1或2
                        # 第i个样本attr属性值的预测值
                        pred = np.argmax(out_dict[attr][i].data.cpu().numpy())
                        # labels[attr_idx]：batch_size个样本的attr属性
                        # labels[attr_idx].data.cpu().numpy()[i]：取出第i个样本的attr属性label值
                        # 第i个样本每个属性值的label值
                        true_label = labels[attr_idx].data.cpu().numpy()[i]
                        # 判断第i个样本的预测值和label值是否相等
                        if pred == true_label:
                            # 如果相等则表示第i个样本的attr属性预测正确，该属性预测正确数加1
                            correct_dict[attr] = correct_dict[attr] + 1
                        # 记录混淆矩阵
                        if attr != "style":
                            if pred == 0:
                                if true_label == 0:
                                    valid_confusion_matrix_record_temp[attr]["TP"] = valid_confusion_matrix_record_temp[attr]["TP"] + 1
                                if true_label == 1:
                                    valid_confusion_matrix_record_temp[attr]["TN"] = valid_confusion_matrix_record_temp[attr]["TN"] + 1
                            if pred == 1:
                                if true_label == 0:
                                    valid_confusion_matrix_record_temp[attr]["FP"] = valid_confusion_matrix_record_temp[attr]["FP"] + 1
                                if true_label == 1:
                                    valid_confusion_matrix_record_temp[attr]["FN"] = valid_confusion_matrix_record_temp[attr]["FN"] + 1

        # 用于记录平均准确率 每个epoch后所有样本所有属性的总准确率 / 属性数
        valid_average_acc = 0
        # 计算每个epoch后每个属性的准确率(80 --> 80%)
        for attr in selectedAttrs:
            correct_dict[attr] = correct_dict[attr] * 100 / (len(test_loader) * batch_size)
            # 记录验证每个epoch每个属性准确率
            valid_acc_attr_record[attr].append(correct_dict[attr])
            valid_average_acc += correct_dict[attr]
        valid_average_acc /= len(selectedAttrs)

        # ---------- Valid End ----------

        # ---------- Chase Best Model Start ----------

        # 比较正确率并保存最佳模型
        if valid_average_acc > best_acc:
            # 记录混淆矩阵
            valid_confusion_matrix_record = valid_confusion_matrix_record_temp
            # 记录最好模型每个标签的准确率
            best_valid_acc_attr = correct_dict
            best_acc = valid_average_acc
            # 将最好的模型参数保存到指定路径
            torch.save(model.state_dict(), config.save_path)
            print("Epoch: %d/%d, Best_acc: %.4f" % (epoch, Epoches, best_acc))

        # ---------- Chase Best Model End ----------

    return train_step_loss_record, valid_step_loss_record, valid_acc_attr_record, valid_confusion_matrix_record, best_valid_acc_attr



