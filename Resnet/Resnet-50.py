import torch 
import torch.nn as nn 
import torchvision 
import torchvision.transforms as transforms 


device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"

epochs = 10
nums_class = 10 
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
    root='../data',
    train=True,
    transform=transforms,
)

test_data = torchvision.datasets.CIFAR10(
    root='../data',
    train=False,
    transform=transforms,
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_data,
    batch_size=batch_size,
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_data,
    batch_size=batch_size,
)


def conv1x1(inplanes, outplanes, stride=1):
    return nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False)

def conv3x3(inplanes, outplanes, stride=1):
    return nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv7x7(inplanes, outplanes, stride=2):
    return nn.Conv2d(inplanes, outplanes, kernel_size=7, stride=stride, padding=3, bias=False)


class BottleNeck(nn.Module):
    expansion = 4 
    def __init__(self, inplanes, outplanes, stride=1, downsample=None, norm_layer=None):
        super(BottleNeck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        self.conv1 = conv1x1(inplanes, outplanes, stride)  #先进行降采样
        self.bn1   = norm_layer(outplanes)
        self.relu  = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(outplanes, outplanes)
        self.bn2   = norm_layer(outplanes)
        self.relu  = nn.ReLU(inplace=True)

        self.conv3 = conv1x1(outplanes, outplanes*self.expansion)
        self.bn1   = norm_layer(outplanes*self.expansion)
        self.relu  = nn.ReLU(inplace=True)

        self.downsample = downsample 
        self.stride = stride 

    def forward(self, x):
        identity = x 
        print("layerx:input",x.shape)
        out = self.conv1(x)
        print("layerx:conv1:",out.shape)
        out = self.bn1(out)
        out = self.relu(out)
        print("layerx:conv1",out.shape)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        print("layerx:conv2",out.shape)

        out = self.conv3(out)
        out = self.bn1(out)
        out = self.relu(out)
        print("layerx:conv3",out.shape)

        if self.downsample:
            identity = self.downsample(x)
        
        out += identity 
        out = self.relu(out)

        return out 


class Resnet50(nn.Module):
    def __init__(self, block, layers, num_class=10):
        super(Resnet50, self).__init__() 

        self.inplanes = 64 

        #经过一个7x7的卷积网络，特征图尺寸减半 
        self.conv1  = conv7x7(3, self.inplanes)
        self.bn1    = nn.BatchNorm2d(self.inplanes)
        self.relu   = nn.ReLU(inplace=True)

        #经过一个最大池化，特征图尺寸减半
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        #残差网络组部分 
        self.layer1  = self.make_layer(block, 64, layers[0], 1)
        self.layer2  = self.make_layer(block, 128, layers[1], 2) 
        self.layer3  = self.make_layer(block, 256, layers[2], 2)
        self.layer4  = self.make_layer(block, 512, layers[3], 2)

        self.avg     = nn.AdaptiveAvgPool2d((1,1))
        self.fc      = nn.Linear(2048*block.expansion, num_class)

    
    def make_layer(self, block, outplanes, blocks, stride=1):
        downsample = None 
        if (stride != 1) or (self.inplanes != outplanes*block.expansion):
            downsample = nn.Sequential(
                conv1x1(self.inplanes, outplanes*block.expansion, stride=stride),
                nn.BatchNorm2d(outplanes*block.expansion)
            )
    
        layers = []
        #先添加一个进行降采样 
        layers.append(block(inplanes=self.inplanes, outplanes=outplanes, stride=stride, downsample=downsample))

        self.inplanes = outplanes * block.expansion

        #后面的len(blocks) - 1 个块都不需要降采样了
        for i in range(1,blocks):
            layers.append(block(self.inplanes, outplanes))

        return nn.Sequential(*layers) 

    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        print("7x7",out.shape)

        out = self.maxpool(out)
        print("maxpool",out.shape)

        out = self.layer1(out)
        print("layer1",out.shape)
        out = self.layer2(out)
        print("layer2",out.shape)
        out = self.layer3(out)
        print("layer3",out.shape)
        out = self.layer4(out)
        print("layer4",out.shape)

        out = self.avg(out)
        out = out.reshape(out.size(0), -1)

        out = self.fc(out)

        return out 


model = Resnet50(BottleNeck, [3, 4, 6, 3]).to(device)

criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr 

steps = len(train_loader)
curr_lr = lr 

for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

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