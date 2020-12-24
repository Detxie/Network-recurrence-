import torch 
import torch.nn as nn 
import torchvision 
import torchvision.transforms as transforms 

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"

epochs = 10 
num_class = 10 
batch_size = 32 
lr = 0.001

transforms = transforms.Compose(
    [
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
    ]
)

train_data = torchvision.datasets.CIFAR10(
    root="../data",
    train=True,
    transform=transforms,
)

test_data = torchvision.datasets.CIFAR10(
    root="../data",
    train=False,
    transform=transforms,
)

train_loader = torch.utils.data.DataLoader(

    dataset=train_data,
    batch_size=batch_size,
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_data,
    batch_size = batch_size,
    shuffle=True,
)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv7x7(in_planes, out_planes, stride=2):
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride, padding=3, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



# N = [(W - F + 2P) / S ]   + 1 
class BasicBlock(nn.Module):

    expansion = 1 #在BasicBlock中值为1，在Boottleneck中值为4
    def __init__(self, in_planes, out_planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__() 

        if norm_layer is None:   #选择批处理函数
            norm_layer = nn.BatchNorm2d
        
        #BasicBlock块 两个3x3的卷积块
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1   = norm_layer(out_planes)
        self.relu  = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_planes, out_planes) #这里不填stride，默认用1，要不然会继续降采样，和conv1一起进行了两次降采样
        self.bn2   = norm_layer(out_planes)
        self.relu  = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride 

    def forward(self, x):
        identity = x 
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample:
            identity = self.downsample(x)
        out += identity   #如果降采样不成功，这里无法实现
        out = self.relu(out)

        return out 


class Resnet34(nn.Module):
    def __init__(self, block, layers, num_class=10):
        super(Resnet34, self).__init__()

        
        #1   先经过一个7x7的卷积,此时特征图尺寸大小减半
        self.in_planes   = 64 
        self.conv1      = conv7x7(3, self.in_planes)
        self.bn1        = nn.BatchNorm2d(self.in_planes)
        self.relu       = nn.ReLU(inplace=True)
        
        #2  经过一个最大池化的过程 ,特征图尺寸大小不变
        self.maxpool1   = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 

        #3  layers[3,4,6,3] ,卷积块的个数分别未3,4,6,3
        self.layer1     = self.make_layer(block, 64, layers[0], 1)  #特征图尺寸减半
        self.layer2     = self.make_layer(block, 128, layers[1], 2)  #特征图尺寸减半
        self.layer3     = self.make_layer(block, 256, layers[2], 2)  #特征图尺寸减半
        self.layer4     = self.make_layer(block, 512, layers[3], 2)  #特征图尺寸减半


        #4
        self.avg        = nn.AdaptiveAvgPool2d((1,1))  #最大池化
        self.fc         = nn.Linear(512 * block.expansion, num_class) #全连接层分类 
    
    def make_layer(self, block, out_planes, blocks, stride=1):
        downsample = None 
        if (stride != 1) or (self.in_planes != out_planes*block.expansion):
            downsample = nn.Sequential(
                conv1x1(self.in_planes, out_planes*block.expansion, stride=stride),
                nn.BatchNorm2d(out_planes*block.expansion),
            )

        layers = [] 
        layers.append(block(self.in_planes, out_planes, stride, downsample))

        self.in_planes = out_planes * block.expansion

        for i in range(1,blocks):
            layers.append(block(self.in_planes, out_planes))
        
        return nn.Sequential(*layers)


    def forward(self, x):
        # print("raw_img",x.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # print("7x7",out.shape)
        out = self.maxpool1(out) 
        # print("maxpool",out.shape)
        out = self.layer1(out)
        # print("layer1",out.shape)
        out = self.layer2(out)
        # print("layer2",out.shape)
        out = self.layer3(out)
        # print("layer3",out.shape)
        out = self.layer4(out)
        # print("layer4",out.shape)

        out = self.avg(out)
        out = out.reshape(out.size(0), -1)

        out = self.fc(out)

        return out 

model  = Resnet34(BasicBlock, [3, 4, 6, 3],).to(device)


criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr  

steps = len(train_loader)
curr_lr  = lr 

for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss    = criterion(outputs, labels)

        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step() 

        if (i+1) % 100 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                   .format(epoch+1, epochs, i+1, steps, loss.item()))

model.eval() 

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'resnet34.ckpt')