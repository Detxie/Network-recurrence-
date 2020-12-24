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


transform = transforms.Compose(
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
    transform=transform,
)

test_data = torchvision.datasets.CIFAR10(
    root="../data",
    train=True,
    transform=transforms.ToTensor()
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_data,
    batch_size=batch_size,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_data,
    batch_size=batch_size,
    shuffle=True,
)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv5x5(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride, padding=2, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.ConV2D(in_planes, out_planes, kernel_size=1, stride=stride)

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1   = norm_layer(out_planes)
        self.relu  = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2   = norm_layer(out_planes)
        self.relu  = nn.ReLU(inplace=True)

        self.downsample = downsample 

    def forward(self, x):
        identity = x 

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            identity = self.downsample(x)
        
        out = out + identity 
        out = self.relu(out)

        return out 

class Resnet(nn.Module):
    def __init__(self, block, layers, num_class=10):
        super(Resnet, self).__init__()

        self.inplanes = 16 
        self.conv   = conv3x3(3,16)
        self.bn     = nn.BatchNorm2d(16)
        self.relu   = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc     = nn.Linear(64, num_class)
    
    def make_layer(self, block, out_planes, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.inplanes != out_planes):  #降采样
            downsample = nn.Sequential(
               conv3x3(self.inplanes, out_planes, stride=stride),
               nn.BatchNorm2d(out_planes)
            )
        layers = [] 
        layers.append(block(self.inplanes, out_planes, stride, downsample))

        self.inplanes = out_planes

        for i in range(1, blocks):
            layers.append(block(out_planes, out_planes))
        
        return nn.Sequential(*layers)

    def forward(self, x ):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.reshape(out.size(0), -1)  #拉成一个常向量 
        out = self.fc(out)
        return out  

model = Resnet(BasicBlock, [2,2,2]).to(device)

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
torch.save(model.state_dict(), 'resnet.ckpt')