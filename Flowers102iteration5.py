import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

epochs = 1000

trainingTransform = transforms.Compose(
    [transforms.Resize(448),
     transforms.RandomResizedCrop(224,(0.5, 1.0)),
     transforms.CenterCrop(224),
     transforms.RandomRotation(20),
     transforms.RandomHorizontalFlip(0.5),
     transforms.RandomVerticalFlip(0.2),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testTransform = transforms.Compose(
    [transforms.Resize(224),
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
    print("training accuracy: "+str((correct/total)*100)+"%")

class NeuralN(nn.Module):
    def __init__(self):
        super().__init__()
        self.Linear1 = nn.Linear(244824, 2048)
        self.Linear2 = nn.Linear(2048, 102)
        self.Pool = nn.AvgPool3d((1, 3, 3),(1, 1, 1))
        self.Dropout = nn.Dropout(0.003)
        self.Conv1 = nn.Conv2d(3, 6, 5, 1)
        self.Conv2 = nn.Conv2d(6, 8, 5, 1)
        self.Conv3 = nn.Conv2d(8, 9, 5, 1)
        self.Conv4 = nn.Conv2d(9, 8, 5, 1)
        self.Conv5 = nn.Conv2d(8, 6, 5, 1)
        self.BatchN1 = nn.BatchNorm2d(6)
        self.BatchN2 = nn.BatchNorm2d(8)
        self.BatchN3 = nn.BatchNorm2d(9)
        self.BatchN4 = nn.BatchNorm2d(8)
        self.BatchN5 = nn.BatchNorm2d(6)

    def forward(self, x):
        x = self.Pool(x)
        x = self.Conv1(x)
        x = F.relu(self.BatchN1(x))
        x = self.Dropout(x)
        x = self.Conv2(x)
        x = F.relu(self.BatchN2(x))
        x = self.Dropout(x)
        x = self.Conv3(x)
        x = F.relu(self.BatchN3(x))
        x = self.Dropout(x)
        x = self.Conv4(x)
        x = F.relu(self.BatchN4(x))
        x = self.Dropout(x)
        x = self.Conv5(x)
        x = F.relu(self.BatchN5(x))
        x = self.Dropout(x)
        x = x.reshape(-1, 244824)
        x = F.relu(self.Linear1(x))
        x = self.Dropout(x)
        x = self.Linear2(x)
        return x

neuNet = NeuralN().to("cuda")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(neuNet.parameters(), 0.0001)

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

    if epoch % 25 == 0:
        validAccuracyFunction()
        trainAccuracyFunction()
        name = './Flowers5-'+str(epoch)+'.pth'
        torch.save(neuNet.state_dict(), name)

print('Finished Training')
torch.save(neuNet.state_dict(), './Flowers5.pth')