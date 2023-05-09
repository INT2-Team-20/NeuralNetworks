import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler

epochs = 3000

trainingTransform = transforms.Compose(
    [
     transforms.RandomResizedCrop((224,224),(0.5, 1.0)),
     transforms.RandomRotation(30),
     transforms.RandomHorizontalFlip(0.5),
     transforms.RandomVerticalFlip(0.2),
     transforms.ColorJitter(brightness=(0.85,1.15),contrast=(0.85,1.15),saturation=(0.85,1.15)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testTransform = transforms.Compose(
    [
     transforms.Resize(224),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainingData = torchvision.datasets.Flowers102(
    root="data",
    split="train",
    download=True,
    transform=trainingTransform,
)

testData = torchvision.datasets.Flowers102(
    root="data",
    split="test",
    download=True,
    transform=testTransform,
)

validData = torchvision.datasets.Flowers102(
    root="data",
    split="val",
    download=True,
    transform=testTransform,
)

trainingDataloader = torch.utils.data.DataLoader(trainingData, 24, True)
testDataloader = torch.utils.data.DataLoader(testData, 24, True)
validDataloader = torch.utils.data.DataLoader(validData, 24, True)
def testAccuracyFunction():
  with torch.no_grad():
    total = 0
    correct = 0
    for data in testDataloader:
        images, labels = data
        images = images.to("cuda")
        labels = labels.to("cuda")
        outputs = neuNet(images)
        _, predictions = torch.max(outputs, 1)
        for label, prediction in zip(labels, predictions):
            total = total + 1
            if label == prediction:
                correct = correct + 1
    global testAccuracy
    testAccuracy = (correct/total)*100
    print("test accuracy: "+str((correct/total)*100)+"%")

def trainAccuracyFunction():
  with torch.no_grad():
    total = 0
    correct = 0
    for data in trainingDataloader:
        images, labels = data
        images = images.to("cuda")
        labels = labels.to("cuda")
        outputs = neuNet(images)
        _, predictions = torch.max(outputs, 1)
        for label, prediction in zip(labels, predictions):
            total = total + 1
            if label == prediction:
                correct = correct + 1
    trainAccuracy = (correct/total)*100
    print("training accuracy: "+str((correct/total)*100)+"%")

def validAccuracyFunction():
  with torch.no_grad():
    total = 0
    correct = 0
    for data in validDataloader:
        images, labels = data
        images = images.to("cuda")
        labels = labels.to("cuda")
        outputs = neuNet(images)
        _, predictions = torch.max(outputs, 1)
        for label, prediction in zip(labels, predictions):
            total = total + 1
            if label == prediction:
                correct = correct + 1
    global validAccuracy
    validAccuracy = (correct/total)*100
    print("valid accuracy: "+str((correct/total)*100)+"%")

class NeuralN(nn.Module):
    def __init__(self):
        super().__init__()
        self.Linear1 = nn.Linear(12800, 2048)
        self.Linear2 = nn.Linear(2048, 102)
        self.Dropout = nn.Dropout(0.125)
        self.Conv1 = nn.Conv2d(3, 64, 3, 1)
        self.BNorm1 = nn.BatchNorm2d(64)
        self.Conv2 = nn.Conv2d(64, 128, 5, 1)
        self.BNorm2 = nn.BatchNorm2d(128)
        self.Conv3 = nn.Conv2d(128, 256, 5, 1)
        self.BNorm3 = nn.BatchNorm2d(256)
        self.Conv4= nn.Conv2d(256, 512, 5, 2)
        self.BNorm4 = nn.BatchNorm2d(512)
        self.Conv5= nn.Conv2d(512, 512, 5, 2)
        self.BNorm5 = nn.BatchNorm2d(512)
        self.Pool = nn.MaxPool2d((2,2),(2,2))
        self.LReLU = nn.LeakyReLU()
        self.PReLU = nn.PReLU()

    def forward(self, x):
        x = self.PReLU(self.BNorm1(self.Conv1(x)))
        x = self.Pool(x)
        x = self.Dropout(x)
        x = self.PReLU(self.BNorm2(self.Conv2(x)))
        x = self.Pool(x)
        x = self.PReLU(self.BNorm3(self.Conv3(x)))
        x = self.PReLU(self.BNorm4(self.Conv4(x)))
        x = self.PReLU(self.BNorm5(self.Conv5(x)))
        x = self.Pool(x)
        x = self.Dropout(x)
        x = torch.flatten(x, start_dim=1)
        x = self.PReLU(self.Linear1(x))
        x = self.Linear2(x)
        return x

neuNet = NeuralN().to("cuda")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(neuNet.parameters(),  0.00005 , weight_decay=0.01) #previous rate:0.00005  , 0.001
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.8)
bestAccuracy = 0
bestTAccuracy = 0
counter = 0
for epoch in range(epochs):
    print(epoch)
    for i, (inputs,labels) in enumerate(trainingDataloader):
        inputs = inputs.to("cuda")
        labels = labels.to("cuda")
        optimizer.zero_grad()
        outputs = neuNet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    scheduler.step()
    if ((epoch < 300) and (epoch % 50) == 0):
        testAccuracyFunction()
        if testAccuracy > bestTAccuracy:
            bestTAccuracy = testAccuracy
            name = './Flowers12-'+str(epoch)+'-'+str(round(testAccuracy))+'.pth'
            torch.save(neuNet.state_dict(), name)

    if ((epoch > 300) and (epoch % 18) == 0):
          testAccuracyFunction()
          if testAccuracy > bestTAccuracy:
            counter = 5
            bestTAccuracy = testAccuracy
            name = './Flowers12-'+str(epoch)+'-'+str(round(testAccuracy))+'.pth'
            torch.save(neuNet.state_dict(), name)

    if counter > 0:
        testAccuracyFunction()
        if testAccuracy > bestTAccuracy:
            counter = 5
            bestTAccuracy = testAccuracy
            name = './Flowers12-'+str(epoch)+'-'+str(round(testAccuracy))+'.pth'
            torch.save(neuNet.state_dict(), name)
        else:
            counter = counter - 1