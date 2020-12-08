import torch  
import torch.nn as nn 
import torchvision 
import torchvision.transforms as transforms 


device = "cpu"  #默认cpu
if torch.cuda.is_available():  #如果环境配好cuda,使用0号显卡
    device = "cuda:0"  



#超惨配置
epochs = 10 
num_class = 10 
batch_size = 128 
lr = 0.001


#构造数据读取器 

train_data = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True,
    transform=transforms.ToTensor(),
    download=True  
)

test_data = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    transform=transforms.ToTensor(),
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_data,
    batch_size=batch_size,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_data,
    batch_size=batch_size,
    shuffle=False, 
)


def conv3x3(inplanes, outplanes, stride=1, padding=1):
    return nn.Conv2d(in_channels=inplanes, out_channels=outplanes, kernel_size=3, stride=stride, padding=padding)

def conv5x5(inplanes, outplanes, stride=1, padding=2):
    return nn.Conv2d(in_channels=inplanes, out_channels=outplanes, kernel_size=5, stride=stride, padding=padding)

class mynet(nn.Module):

    #output 
    #N = [(W - F + 2P) / S ]   + 1 
    # N:输出后图像大小
    # W:原图像大小
    # F:卷积核大小
    # P:padding值的大小
    # S:stride值的大小  
    def __init__(self, num_class=1000, norm_layer=None):
        super(mynet, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        #stage1 
        self.conv1 = conv5x5(1, 32, stride=1, padding=2)  #size:(28-5+4)/1 + 1 = 28
        self.bn1   = norm_layer(32)
        self.relu  = nn.ReLU(inplace=True)

        #stage2
        self.conv2 = conv5x5(32, 64, stride=1, padding=2)  #size = 28
        self.bn2   = norm_layer(64)
        self.relu  = nn.ReLU(inplace=True)


        #stage3
        self.conv3 = conv3x3(64, 128, stride=1, padding=1)  #size = 28*28
        self.bn3   = norm_layer(128)
        self.relu  = nn.ReLU(inplace=True)

        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)  #size减半 14*14

        self.fc    = nn.Linear(14*14*128, num_class)


    def forward(self, x):
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.pool(out)
        
        out = out.reshape(out.size(0), -1)  #将整个卷积拉成一个长串 32*14*14*128 size(0) = 32 
        
        out = self.fc(out)

        return out 

model  = mynet(num_class).to(device)

net_loss        = nn.CrossEntropyLoss()
optimizer   = torch.optim.Adam(model.parameters(), lr=lr)

#train 

steps = len(train_loader)

for epoch in range(epochs):
    for i ,(images, labels) in enumerate(train_loader):
        images  = images.to(device)
        labels  = labels.to(device)

        outputs = model(images)
        loss    = net_loss(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, epochs, i+1, steps, loss.item()))

model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
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

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), '/mynet/model.ckpt')