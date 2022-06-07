import numpy as np

from train import train
from utils import loss_curve, acc_curve, acc_cur

# 开始训练
train_step_loss_record, valid_step_loss_record, valid_acc_attr_record, valid_confusion_matrix_record, best_valid_acc_attr = train()
# 训练集和验证集上的损失
# loss_curve(train_step_loss_record, valid_step_loss_record)
# 修改
# colorList = ["red", "blue", "green", "yellow", "purple", "black"]
colorList = ["red"]
# 各个属性的精确度
# acc_curve(valid_acc_attr_record, colorList)
# 多标签平均精度
average = np.empty(len(valid_acc_attr_record["hair"]))
for i in range(len(valid_acc_attr_record["hair"])):
    average[i] = (valid_acc_attr_record["hair"][i] + valid_acc_attr_record["gender"][i] + valid_acc_attr_record["earring"][i] + valid_acc_attr_record["smile"][i] + valid_acc_attr_record["frontal_face"][i] + valid_acc_attr_record["style"][i]) / 6
acc_cur(average)


print("============================")
print("混淆矩阵：", valid_confusion_matrix_record)
print("============================")
print("Acc：", best_valid_acc_attr)
