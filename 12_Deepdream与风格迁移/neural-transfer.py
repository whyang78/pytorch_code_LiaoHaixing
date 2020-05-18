import torch
from torch import nn,optim
from torch.nn import functional as F
from torchvision import transforms,models
from PIL import Image
from matplotlib import  pyplot as plt
import copy

device=torch.device('cuda:0')
img_h=512
img_w=600
tf=transforms.Compose([
    transforms.Resize((img_h,img_w)),
    transforms.ToTensor(),
])

def load_image(path):
    img = Image.open(path).convert('RGB')
    img_tensor = tf(img).unsqueeze(0)
    return img_tensor.to(device)

def show_image(img_tensor):
    img=img_tensor.cpu().clone().squeeze(0)
    t=transforms.ToPILImage()
    img_=t(img)
    plt.imshow(img_)
    plt.show()

class contentLoss(nn.Module):
    def __init__(self,target):
        super(contentLoss, self).__init__()
        self.target=target.detach()

    def forward(self, x):
        self.loss=F.mse_loss(x,self.target)
        return x

def gram_martrix(x):
    b,c,h,w=x.size()
    x_=x.view(b*c,h*w)
    gram=torch.mm(x_,x_.t())
    return gram.div(b*c*h*w)

class styleLoss(nn.Module):
    def __init__(self,target):
        super(styleLoss, self).__init__()
        self.target=gram_martrix(target).detach()

    def forward(self,x):
        gram=gram_martrix(x)
        self.loss=F.mse_loss(gram,self.target)
        return x

cnn=models.vgg19(pretrained=True).features.to(device).eval()
# import pprint
# # pprint.pprint(list(cnn.children()))

class Normalization(nn.Module):
    def __init__(self, mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().view(1,3, 1, 1).to(device)
        self.std = std.clone().detach().view(1,3, 1, 1).to(device)

    def forward(self, img):
        return (img - self.mean) / self.std

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
def get_style_model_and_losses(cnn,style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn=copy.deepcopy(cnn)
    model=nn.Sequential(Normalization().to(device))
    content_losses=[]
    style_losses=[]

    i=0
    for layer in cnn.children():
        if isinstance(layer,nn.Conv2d):
            i+=1
            name='conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name,layer)

        if name in content_layers:
            content_target=model(content_img).detach()
            content_loss=contentLoss(content_target)
            model.add_module('content_loss_{}'.format(i),content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            style_target=model(style_img).detach()
            style_loss=styleLoss(style_target)
            model.add_module('style_loss_{}'.format(i),style_loss)
            style_losses.append(style_loss)

    for i in range(len(model)-1,-1,-1):
        if isinstance(model[i],contentLoss) or isinstance(model[i],styleLoss):
            break
    model=model[:(i+1)]
    return model,content_losses,style_losses

# 一般style_weight比较大
def run_style_transfer(cnn, content_img, style_img, input_img, num_steps=600,
                       style_weight=1000000, content_weight=1):
    model, content_losses, style_losses=get_style_model_and_losses(cnn,style_img,content_img)
    optimizer=optim.LBFGS([input_img.requires_grad_()])

    run=[0]
    while run[0]<=num_steps:
        def closure():
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)

            style_loss=0
            content_loss=0
            for sl in style_losses:
                style_loss+=sl.loss
            for cl in content_losses:
                content_loss+=cl.loss
            loss=style_weight*style_loss+content_weight*content_loss
            loss.backward()

            run[0]+=1
            if run[0] % 50 == 0:
                print("epoch {}:".format(run[0]))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_loss.item(), content_loss.item()))
                print()
            return loss
        optimizer.step(closure)

    input_img.data.clamp_(0, 1)
    return input_img


content_path='./content.png'
content_img=load_image(content_path)
style_path='./starry_night.jpg'
style_img=load_image(style_path)
input_img = content_img.clone() #直接用原图片
# input_img = torch.randn(content_img.data.size(), device=device) #随机生成也可以
output = run_style_transfer(cnn, content_img, style_img, input_img)
show_image(output)