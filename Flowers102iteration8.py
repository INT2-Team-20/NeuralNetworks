import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

epochs = 1000

trainingTransform = transforms.Compose(
    [transforms.Resize(448),
     transforms.RandomRotation(20),
     transforms.RandomResizedCrop((224,224),(0.5, 1.0)),
     transforms.RandomHorizontalFlip(0.5),
     transforms.RandomVerticalFlip(0.3),
     transforms.ColorJitter(0.3, 0.025),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testTransform = transforms.Compose(
    [transforms.Resize((224,224)),
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
        self.Linear1 = nn.Linear(1800, 2048)
        self.Linear2 = nn.Linear(2048, 102)
        self.Dropout = nn.Dropout(0.02)
        self.Conv1 = nn.Conv2d(3, 150, 5, 2)
        self.BNorm1 = nn.BatchNorm2d(150)
        self.Conv2 = nn.Conv2d(150, 300, 5, 2)
        self.BNorm2 = nn.BatchNorm2d(300)
        self.Conv3 = nn.Conv2d(300, 450, 5, 2)
        self.BNorm3 = nn.BatchNorm2d(450)
        self.Pool = nn.MaxPool2d((2,2),(2,2))
        self.LReLU = nn.LeakyReLU()

    def forward(self, x):
        x = self.LReLU(self.BNorm1(self.Conv1(x)))
        x = self.Dropout(x)
        x = self.Pool(x)
        x = self.LReLU(self.BNorm2(self.Conv2(x)))
        x = self.Pool(x)
        x = self.LReLU(self.BNorm3(self.Conv3(x)))
        x = self.Dropout(x)
        x = self.Pool(x)
        x = x.reshape(-1, 1800)
        x = self.LReLU(self.Linear1(x))
        x = self.Dropout(x)
        x = self.Linear2(x)
        return x

neuNet = NeuralN().to("cuda")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(neuNet.parameters(), 0.00005)
bestAccuracy = 0
bestTAccuracy = 0
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

    if epoch % 20 == 0:
        validAccuracyFunction()
        trainAccuracyFunction()
        if validAccuracy > bestAccuracy:
          bestAccuracy = validAccuracy
          testAccuracyFunction()
          if testAccuracy > bestTAccuracy:
            bestAccuracy = bestTAccuracy
            name = './Flowers8-'+str(epoch)+'.pth'
            torch.save(neuNet.state_dict(), name)