import torch
import models_vit
import torch.nn as nn
import os
import numpy as np
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from torch.utils.data import DataLoader
from loaddata import CrossPairedDataset, get_train_val_test_loaders

script_dir = os.path.dirname(os.path.abspath(__file__))
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import seaborn as sns

import tensorflow as tf
from torch.optim import Adam

from CCAtools import cca_loss


class MlpNet(nn.Module):
    def __init__(self, layer_sizes, input_size):
        super(MlpNet, self).__init__()
        layers = []
        layer_sizes = [input_size] + layer_sizes
        for l_id in range(len(layer_sizes) - 1):
            if l_id == len(layer_sizes) - 2:
                layers.append(nn.Sequential(
                    nn.BatchNorm1d(num_features=layer_sizes[l_id], affine=False),
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                ))
            else:
                layers.append(nn.Sequential(
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                    nn.Sigmoid(),
                    nn.BatchNorm1d(num_features=layer_sizes[l_id + 1], affine=False),
                ))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        # print('111',x.type)
        for layer in self.layers:
            x = layer(x)
        # print('222',x.type)
        return x


class DeepCCA(nn.Module):
    def __init__(self, layer_sizes1, layer_sizes2, input_size1, input_size2, outdim_size, use_all_singular_values, device=torch.device('cpu')):
        super(DeepCCA, self).__init__()
        self.model1 = MlpNet(layer_sizes1, input_size1)
        self.model2 = MlpNet(layer_sizes2, input_size2)

        self.loss = cca_loss(outdim_size, use_all_singular_values, device).loss

    def forward(self, x1, x2):
        """

        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]

        """
        # feature * batch_size
        output1 = self.model1(x1)
        output2 = self.model2(x2)

        return output1, output2

def prepare_model(chkpt_dir, arch='vit_large_patch16'):
       # build model
    model = models_vit.__dict__[arch](
        img_size=224,
        num_classes=4,
        drop_path_rate=0.2,
        global_pool=False,
    )
    # load model
    # load RETFound weights
    print(f"Loading checkpoint from: {chkpt_dir}")
    checkpoint = torch.load(chkpt_dir, map_location='cuda')
    checkpoint_model = checkpoint['model']

    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    model.load_state_dict(checkpoint_model, strict=False)

    # 冻结模型权重
    # print("Freezing all model parameters...")
    # for param in model.parameters():
    #     param.requires_grad = False

    # return model
    from LoRA.lora import LoRA_ViT_timm

    lora_vit_oct = LoRA_ViT_timm(vit_model=model, r=2, alpha=2, num_classes=4)

    return lora_vit_oct

# 设定整体架构
class CombinedModel(nn.Module):
    def __init__(self, model_oct_path, model_cfp_path, arch, combined_out_features, num_classes,deepcca_outdim_size):
        super(CombinedModel, self).__init__()
        print("Initializing Models...")
        self.model_oct = prepare_model(model_oct_path, arch)
        self.model_cfp = prepare_model(model_cfp_path, arch)

        input_size1 = 1024
        input_size2 = 1024
        self.deepcca = DeepCCA(layer_sizes1=[1024, deepcca_outdim_size], layer_sizes2=[1024, deepcca_outdim_size],
                input_size1=input_size1, input_size2=input_size2,
                outdim_size=deepcca_outdim_size, use_all_singular_values=True)

        # 添加池化层以调整特征图尺寸
#         self.pool = nn.AdaptiveAvgPool2d((224, 224))
        print("Initializing fully connected layers...")
        self.fc1 = nn.Linear(deepcca_outdim_size * 2, combined_out_features) # 每个模型的输出是1024维
        self.fc2 = nn.Linear(combined_out_features, num_classes)
        self.relu = nn.ReLU()

    ##新加的
    def to(self,device):
        super().to(device)
        self.deepcca.model1.to(device)
        self.deepcca.model2.to(device)
        return self

    def forward(self, x_oct, x_cfp):
        # print("Processing OCT features...")
        x_oct_features = self.model_oct.forward_features(x_oct.float())
        x_oct_features = torch.mean(x_oct_features, dim=1)

        #  mean_features 的形状是 [32, 1024]

        # print("Processing CFP features...")
        x_cfp_features = self.model_cfp.forward_features(x_cfp.float())
        x_cfp_features = torch.mean(x_cfp_features, dim=1)


        # x_combined = torch.stack((x_deepcca_features1, x_deepcca_features2), dim=2)

#         x_combined = np.stack((x_oct_features, x_cfp_features), axis=1).reshape(1024, 2)
        x_deepcca_features1, x_deepcca_features2 = self.deepcca(x_oct_features.float(), x_cfp_features.float())
        # 合并特征
        x_combined = torch.cat((x_deepcca_features1, x_deepcca_features2), dim=1)

#         print(x_combined.shape)
#         x_combined = x_combined.reshape(1024, 2)
        # print("Passing combined features through the network...")
        x = self.fc1(x_combined)
        x = self.relu(x)
        x = self.fc2(x)
        return x

##scikit-learn、classification_report函数并不直接提供特异性(specificity)的计算
def calculate_specificity(cm,class_index):
    tp=cm[class_index,class_index]
    fn=cm[class_index,:].sum() - tp
    fp=cm[:,class_index].sum()-tp
    tn=cm.sum()-(tp+fn+fp)
    specificity=tn/(tn+fp)
    return specificity
def calculate_overall_specificity(cm):
    specificities=[]
    for class_index in range(len(cm)):
        specificities.append(calculate_specificity(cm,class_index))
    return sum(specificities)/len(specificities)

def test_model(model, test_loader,device = None,savepath =  os.path.join(script_dir,'combineCCALoRAEarly_model.pth')):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model.to(device)
    model.load_state_dict(torch.load(savepath))
    model.eval()  # Set model to evaluation mode

    y_true = []
    y_pred = []
    y_pred_probs = []
    # 出T-SNE
    all_features = []
    all_labels = []

    print("Starting model evaluation...")
    with torch.no_grad():  # Disable gradient calculation禁用梯度计算
        for batch_idx, (oct_imgs, cfp_imgs, labels) in enumerate(test_loader):
            print(f"Batch {batch_idx+1}/{len(test_loader)}")
            outputs = model(oct_imgs.to(device), cfp_imgs.to(device))
            _, predicted = torch.max(outputs, dim=1)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_pred_probs.extend(probabilities.cpu().numpy())

            all_features.append(outputs.detach().cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    #测试集T-SNE图
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    ##T-SNE可视化
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(all_features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=all_labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, ticks=np.arange(4))
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE Visualization of Four-Class Data')
    plt.savefig("T-SNE_Test.png")
    plt.show()

    # Calculate evaluation metrics 计算评估指标
    print("Calculating evaluation metrics...")
    from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score,classification_report,confusion_matrix

    #使用classification_report生成分类报告
    target_names=['a_nor','b_dry','c_pcv','d_wet']
    report=classification_report(y_true,y_pred,target_names=target_names,output_dict=True)

    # # 测试集混淆矩阵   使用confusion生成的  归一化之前的
    # cm = confusion_matrix(y_true, y_pred)
    # labels = ["a_nor", "b_dry", "c_pcv", "d_wet"]
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=labels, yticklabels=labels)
    # plt.title("Confusion Matrix")
    # plt.xlabel("Predicted Labels")
    # plt.ylabel("True Labels")
    # plt.savefig("confusion_matrix_test.png")
    #
    # plt.show()  # test测试集混淆矩阵

    #测试集混淆矩阵   使用confusion生成的 归一化之后的
    cm= confusion_matrix(y_true,y_pred)
    cm_normalized = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
    labels=["a_nor","b_dry","c_pcv","d_wet"]
    plt.figure(figsize=(8,6))
    sns.heatmap(cm_normalized,annot=True,cmap="Blues",fmt=".4f",xticklabels=labels,yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.savefig("confusion_matrix_test.png")

    plt.show()  #test测试集混淆矩阵

    #打印分类报告
    print("Classification Report.....\n")
    for label,metrics in report.items():
        if label in target_names:
            print(f"Class{label}:")
            print(f"Recall(Sensitivity):{metrics['recall']:.4f}")
            print(f"Specificity:{calculate_specificity(cm, target_names.index(label)):.4f}")
            print(f"F1-score:{metrics['f1-score']:.4f}")
            print(f"Precision:{metrics['precision']:.4f}")

     #打印总体指标
    print("\nOverall Metrics:")
    print(f"Test Recall(Sensitivity):{report['macro avg']['recall']:.4f}")
    print(f"Test Specificity:{calculate_overall_specificity(cm):.4f}")
    print(f"Test F1-score:{report['macro avg']['f1-score']:.4f}")
    print(f"Test Precision:{report['macro avg']['precision']:.4f}")

       ###y_pred是类别标签，y_pred_probs是类别概率
    accuracy = accuracy_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_pred_probs, multi_class='ovr')
    auc_pr = average_precision_score(y_true, y_pred_probs, average="macro")
    print(f"Test Accuracy: {accuracy}")
    print(f"Test AUC-ROC: {auc_roc}")
    print(f"Test AUC-PR: {auc_pr}")


def evaluate_model(model, val_loader, device, loss_function):   #计算loss来评估模型的，作用是计算Loss和acc，实时打印
    model.eval()
    total_loss = 0 #总损失
    total_samples = 0 #样本数量
    correct_predictions = 0 #正确预测数量
    y_true = []
    y_pred = []
    y_pred_probs = []

    with torch.no_grad():
        for oct_imgs, cfp_imgs, labels in val_loader:
            oct_imgs, cfp_imgs, labels = oct_imgs.to(device), cfp_imgs.to(device), labels.to(device)
            #出结果
            outputs = model(oct_imgs, cfp_imgs)
            #计算损失
            loss = loss_function(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

            #计算准确率，获取预测和真实值列表
            # 根据outputs出测结果,预测值y_pred和真实值y_true
            _, predicted = torch.max(outputs, dim=1)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_pred_probs.extend(probabilities.cpu().detach().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_pred_probs, multi_class='ovr')
    auc_pr = average_precision_score(y_true, y_pred_probs, average="macro")
    average_loss = total_loss / total_samples
    return average_loss,accuracy,auc_roc,auc_pr,y_true,y_pred

def train_model(model, train_loader, val_loader, test_loader, epochs=100, patience=10, script_dir='', savepath='combineCCALoRAEarly_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = Adam(model.parameters(), lr=0.001,weight_decay=0.001)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001, weight_decay=0.0001)
    loss_function = CrossEntropyLoss()

    savepath = os.path.join(script_dir, savepath)
    best_loss = float('inf')
    epochs_no_improve = 0

  #绘制loss、ac曲线
    log_dir1= "logs_train_loss"
    log_dir2= "logs_val_loss"

    log_dir3= "logs_train_Acc"
    log_dir4= "logs_val_Acc"

    writer1 =SummaryWriter(log_dir=log_dir1)
    writer2 =SummaryWriter(log_dir=log_dir2)
    writer3 =SummaryWriter(log_dir=log_dir3)
    writer4 =SummaryWriter(log_dir=log_dir4)

    #出T-SNE
    all_features=[]
    all_labels=[]

    print("Starting training...")
    model.train()
    for epoch in range(epochs):
        for oct_imgs, cfp_imgs, labels in train_loader:
            oct_imgs, cfp_imgs, labels = oct_imgs.to(device), cfp_imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(oct_imgs, cfp_imgs)
            loss = loss_function(outputs, labels)

            lambda_l1 = 1e-5
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss += lambda_l1 * l1_norm

            loss.backward()
            optimizer.step()

            #提取特征和标签
            all_features.append(outputs.detach().cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        #根据evaluate_model获取各个指标。
        train_loss,train_accuracy,train_auc_roc,train_auc_pr,train_true,train_pred=evaluate_model(model,train_loader,device,loss_function)
        val_loss,val_accuracy,val_auc_roc,val_auc_pr,val_ture,val_pred= evaluate_model(model, val_loader, device, loss_function)
        # test_loss,test_accuracy,test_auc_roc,test_auc_pr,test_ture,test_pred= evaluate_model(model, test_loader, device, loss_function)


        #写入网站，可视化曲线,Loss and Acc
        writer1.add_scalar("Loss/train",train_loss,epoch)
        writer2.add_scalar("Loss/val",val_loss,epoch)
        # writer2.add_scalar("Loss/test",test_loss,epoch)

        writer3.add_scalar("Acc/train",train_accuracy,epoch)
        writer4.add_scalar("Acc/val",val_accuracy,epoch)
        # writer4.add_scalar("Acc/test",test_accuracy,epoch)


        #打印loss和评估指标
        print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Train AUC-ROC: {train_auc_roc}, Train AUC-PR: {train_auc_pr}")
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}, Validation AUC-ROC: {val_auc_roc}, Validation AUC-PR: {val_auc_pr}")
        # print(f"Epoch {epoch+1}, Test Loss: {test_loss}, Test Accuracy: {test_accuracy}, Test AUC-ROC: {test_auc_roc}, Test AUC-PR: {test_auc_pr}")



        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), savepath)
            epochs_no_improve = 0
            print("Validation loss decreased, saving model...")
        else:
            epochs_no_improve += 1
            print("Validation loss did not decrease, patience counter:", epochs_no_improve)
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

        # if test_loss < best_loss:
        #     best_loss = test_loss
        #     torch.save(model.state_dict(), savepath)
        #     epochs_no_improve = 0
        #     print("Test loss decreased, saving model...")
        # else:
        #     epochs_no_improve += 1
        #     print("Test loss did not decrease, patience counter:", epochs_no_improve)
        #     if epochs_no_improve >= patience:
        #         print("Early stopping triggered.")
        #         break



    model.load_state_dict(torch.load(savepath))
    print("Training completed. Best model loaded.")

    print("Class to index mapping:")
    #提取训练期间的所有特征和标签  训练集的
    all_features=np.concatenate(all_features,axis=0)
    all_labels=np.concatenate(all_labels,axis=0)

    ##T-SNE可视化
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(all_features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=all_labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, ticks=np.arange(4))
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE Visualization of Four-Class Data')
    plt.savefig("T-SNE_amd_Visualization.png")
    plt.show()

    # train训练集混淆矩阵
    cm = confusion_matrix(train_true, train_pred)
    labels = ["a_nor", "b_dry", "c_pcv", "d_wet"]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.savefig("confusion_matrix_train.png")

    plt.show()    #train训练集混淆矩阵

    # val验证集混淆矩阵
    cm = confusion_matrix(val_ture, val_pred)
    labels = ["a_nor", "b_dry", "c_pcv", "d_wet"]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.savefig("confusion_matrix_val.png")

    plt.show()  # val验证集混淆矩阵

    test_model(model, test_loader, device=device, savepath=savepath)


transform = transforms.Compose([
transforms.Resize((224, 224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.ToTensor()

# oct_root = os.path.join(script_dir, 'oct_data/train')
# cfp_root =  os.path.join(script_dir, 'cfp_data/train')
# traindataset = CrossPairedDataset(oct_root, cfp_root, mode='pair',transform=transform, device=device,resample=True)
# trainloader = DataLoader(traindataset, batch_size=32, shuffle=True)
#
# oct_root = os.path.join(script_dir, 'oct_data/test')
# cfp_root =  os.path.join(script_dir, 'cfp_data/test')
# testdataset = CrossPairedDataset(oct_root, cfp_root, mode = 'pair',transform=transform, device=device,resample=False)
# testloader = DataLoader(testdataset, batch_size=32, shuffle=True)
#
# oct_root = os.path.join(script_dir, 'oct_data/val')
# cfp_root =  os.path.join(script_dir, 'cfp_data/val')
# valdataset = CrossPairedDataset(oct_root, cfp_root, mode='pair',transform=transform, device=device,resample=False)
# valloader = DataLoader(valdataset, batch_size=32, shuffle=True)

oct_root = os.path.join(script_dir, 'oct_data/all')
cfp_root = os.path.join(script_dir, 'cfp_data/all')
alldataset = CrossPairedDataset(oct_root, cfp_root, mode='pair', transform=transform, device=device)
train_loader, val_loader, test_loader = get_train_val_test_loaders(alldataset, train_ratio=0.7, val_ratio=0.15,
                                                                   batch_size=16, shuffle=True, num_workers=0)
# shuffle=True  在每个epoch内图像会被打乱

model_oct_path = os.path.join(script_dir,'RETFound_oct_weights.pth')
model_cfp_path = os.path.join(script_dir,'RETFound_cfp_weights.pth')
model_type = 'vit_large_patch16'

if __name__=='__main__':
    # 模型参数
    # num_features = 1024  # 假设每个单独的模型输出1024个特征

    # combined_out_features = 128 #中间层
    # num_classes = 4
    # deepcca_outdim_size = 128

    model = CombinedModel(model_oct_path, model_cfp_path, model_type, combined_out_features=256, num_classes=4,deepcca_outdim_size=128)
    model.to(torch.device('cuda'))

    # print("要输出model")
    # print(model)
    # 打印参数
    n_parameters_before = sum(p.numel() for p in model.parameters() if p.requires_grad)  ##p.requires_grad是指在梯度过程中可训练的参数数量
    print('number of params (M): %.2f' % (n_parameters_before / 1.e6))    ##上面的CombinedModel已经用过Lora了，所以参数会减小。

    #实际运行需要替换成
    train_model(model, train_loader, val_loader, test_loader, epochs=100, patience=50, script_dir='', savepath='combineCCALoRAEarly_model.pth')
    #仅测试结果
    #test_model(model, test_loader, device=device, savepath='combineCCALoRAEarly_model.pth')

