import argparse
import os
import cv2
import numpy as np
import torch
import models_vit
import torch.nn as nn
from LoRA.lora import LoRA_ViT_timm
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from CCAtools import cca_loss
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image  ##show_cam_on_image叠加CAM到图像上  preprocess_image用于图像预处理
from pytorch_grad_cam.ablation_layer import AblationLayerVit

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

    model.load_state_dict(checkpoint_model, strict=False)  # strict=False表示模型参数和加载的权重参数不用严格配对

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


        x_cfp_features = self.model_cfp.forward_features(x_cfp.float())
        x_cfp_features = torch.mean(x_cfp_features, dim=1)

        x_deepcca_features1, x_deepcca_features2 = self.deepcca(x_oct_features.float(), x_cfp_features.float())

        x_combined = torch.cat((x_deepcca_features1, x_deepcca_features2), dim=1)

        x = self.fc1(x_combined)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='是否用GPU')
    # parser.add_argument('--image-path',type=str,default='./both.png',
    #                     help='图像路径')
    parser.add_argument('--cfp-folder',type=str,default='./cfp_images',
                        help='CFP图像文件夹路径')
    parser.add_argument('--oct-folder',type=str,default='./oct_images',
                        help='OCT图像文件夹路径')
    parser.add_argument('--output-folder-cfp',type=str,default='./output_cfp',
                        help='CFP图像文件夹路径')
    parser.add_argument('--output-folder-oct',type=str,default='./output_oct',
                        help='OCT图像文件夹路径')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='平滑CAM')
    parser.add_argument('--eigen_smooth',action='store_true',
                        help='通过获取cam_weight*activations的第一成分来减少噪声')
    parser.add_argument('--method',type=str,default='layercam',
                        help='指定方法 gradcam/gradcam++/scorecam/xgradcam/ablationcam/eigencam/eigengradcam/layercam/fullgrad')

    args = parser.parse_args() #解析命令行参数

    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))
    #对输入的张量进行变换，将通道维度移到第一个位置
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


if __name__ == '__main__':
    """ python vit_gradcam.py --image-path <path_to_image>
    Example usage of using cam-methods on a VIT network.
    """
    args = get_args()#解析命令行参数

    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    # model = torch.hub.load('facebookresearch/deit:main','deit_tiny_patch16_224', pretrained=True)
    #使用Pytorch Hub加载预训练的 ViT模型，

    model_oct_path = './RETFound_oct_weights.pth'
    model_cfp_path = './RETFound_cfp_weights.pth'
    arch='vit_large_patch16'
    model = CombinedModel(model_oct_path, model_cfp_path, arch, combined_out_features=256, num_classes=4,deepcca_outdim_size=128)
    print(model)
    model.load_state_dict(torch.load('combineCCALoRAEarly_model.pth'))

    model.eval()#评估模式

    if args.use_cuda:
        model = model.cuda()

    target_layers_cfp = [model.model_oct.lora_vit.blocks[-1].norm1]  #目标层
    target_layers_oct = [model.model_oct.lora_vit.blocks[-1].norm1]  #目标层
    # target_layers = [model.blocks[-1].norm1]  #目标层


    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    if args.method == "ablationcam":
        cam_cfp = methods[args.method](model=model.model_oct,
                                   target_layers=target_layers_oct,
                                   reshape_transform=reshape_transform,
                                   ablation_layer=AblationLayerVit())
        cam_oct = methods[args.method](model=model.model_oct,
                                   target_layers=target_layers_oct,
                                   reshape_transform=reshape_transform,
                                   ablation_layer=AblationLayerVit())
    else:
        cam_cfp = methods[args.method](model=model.model_oct,
                                   target_layers=target_layers_oct,
                                   reshape_transform=reshape_transform)
        cam_oct = methods[args.method](model=model.model_oct,
                                   target_layers=target_layers_oct,
                                   reshape_transform=reshape_transform)

    cfp_images=os.listdir(args.cfp_folder)
    oct_images=os.listdir(args.oct_folder)

    ##两种模态图片配对
    paired_files=[]
    for cfp_image in cfp_images:
        cfp_base = "_".join(cfp_image.split("-")[2:4])
        for oct_image in oct_images:
            oct_base = "_".join(oct_image.split("-")[2:4])
            if cfp_base == oct_base:
                paired_files.append((cfp_image,oct_image))
                break
    #打印配对结果
    print(f"共有{len(paired_files)}对文件:")
    for i,(cfp_image,oct_image) in enumerate(paired_files):
        print(f"配对{i+1}:CFP图像:{cfp_image},OCT图像：{oct_image}")

    for cfp_image,oct_image in paired_files:
        cfp_img_path=os.path.join(args.cfp_folder,cfp_image)
        oct_img_path=os.path.join(args.oct_folder,oct_image)

        cfp_img = cv2.imread(cfp_img_path, 1)[:, :, ::-1]  #读取图像并转换为RGB格式 1是读取rgb
        cfp_img = cv2.resize(cfp_img, (224, 224))  #调整图像大小为224*224
        cfp_img = np.float32(cfp_img) / 255  #归一化图像到[0，1]范围
        cfp_input_tensor = preprocess_image(cfp_img, mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])  #将图像转换为模型输入张量

        oct_img = cv2.imread(oct_img_path, 1)[:, :, ::-1]  #读取图像并转换为RGB格式 1是读取rgb
        oct_img = cv2.resize(oct_img, (224, 224))  #调整图像大小为224*224
        oct_img = np.float32(oct_img) / 255  #归一化图像到[0，1]范围
        oct_input_tensor = preprocess_image(oct_img, mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])  #将图像转换为模型输入张量

        # If None, returns the map for the highest scoring category.
        # Otherwise, targets the requested category.
        targets = None #生成最高得分类别的激活图，如果指定了就是别的

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam_cfp.batch_size = 1024  #CAM批处理大小为32，加快计算速度，适用于AblationLayerVit 和ScoreCAM
        cam_oct.batch_size = 1024  #CAM批处理大小为32，加快计算速度，适用于AblationLayerVit 和ScoreCAM

        grayscale_cam_cfp = cam_cfp(input_tensor=cfp_input_tensor,targets=targets,eigen_smooth=args.eigen_smooth,aug_smooth=args.aug_smooth) #生成CAM图像，使用指定的参数进行平滑和降噪
        grayscale_cam_oct = cam_oct(input_tensor=oct_input_tensor,targets=targets,eigen_smooth=args.eigen_smooth,aug_smooth=args.aug_smooth) #生成CAM图像，使用指定的参数进行平滑和降噪

        # Here grayscale_cam has only one image in the batch
        grayscale_cam_cfp = grayscale_cam_cfp[0, :]  #因为批次中只有一张图像，提取其CAM图
        grayscale_cam_oct = grayscale_cam_oct[0, :]  #因为批次中只有一张图像，提取其CAM图

        cam_image_cfp = show_cam_on_image(cfp_img, grayscale_cam_cfp)  #将CAM图像叠加到原始图像上
        cam_image_oct = show_cam_on_image(oct_img, grayscale_cam_oct)  #将CAM图像叠加到原始图像上

        os.makedirs(args.output_folder_cfp,exist_ok=True)
        os.makedirs(args.output_folder_oct,exist_ok=True)

        cfp_output_path=os.path.join(args.output_folder_cfp,f'cfp_{cfp_image}')
        oct_output_path=os.path.join(args.output_folder_oct,f'oct_{oct_image}')

        cv2.imwrite(cfp_output_path, cam_image_cfp)
        cv2.imwrite(oct_output_path, cam_image_oct)