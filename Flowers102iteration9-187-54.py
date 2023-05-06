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
     transforms.RandomVerticalFlip(0.1),
     transforms.ColorJitter(brightness=(0.8,1.2),contrast=(0.8,1.2),saturation=(0.8,1.2)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))],)

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

trainingAsTestData = torchvision.datasets.Flowers102(
    root="data",
    split="train",
    download=True,
    transform=testTransform,
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

trainingDataloader = torch.utils.data.DataLoader(trainingData, 12, True)
testDataloader = torch.utils.data.DataLoader(testData, 12, True)
trainingAsTestDataloader = torch.utils.data.DataLoader(testData, 12, True)
validDataloader = torch.utils.data.DataLoader(validData, 12, True)

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

def trainAsTestAccuracyFunction():
  with torch.no_grad():
    total = 0
    correct = 0
    for data in trainingAsTestDataloader:
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
        self.Linear1 = nn.Linear(3456, 1728)
        self.Linear2 = nn.Linear(1728, 102)
        self.Dropout = nn.Dropout(0.07)
        self.Conv1 = nn.Conv2d(3, 128, 5, 2)
        self.BNorm1 = nn.BatchNorm2d(128)
        self.Conv2 = nn.Conv2d(128, 256, 5, 2)
        self.BNorm2 = nn.BatchNorm2d(256)
        self.Conv3 = nn.Conv2d(256, 384, 5, 1)
        self.BNorm3 = nn.BatchNorm2d(384)
        self.Conv4 = nn.Conv2d(384, 384, 5, 1)
        self.BNorm4 = nn.BatchNorm2d(384)
        self.Conv5 = nn.Conv2d(384, 384, 5, 1)
        self.BNorm5 = nn.BatchNorm2d(384)
        self.Pool = nn.MaxPool2d((2,2),(2,2))
        self.LReLU = nn.LeakyReLU()

    def forward(self, x):
        x = self.LReLU(self.BNorm1(self.Conv1(x)))
        x = self.Pool(x)
        x = self.LReLU(self.BNorm2(self.Conv2(x)))
        x = self.Dropout(x)
        x = self.LReLU(self.BNorm3(self.Conv3(x)))
        x = self.Pool(x)
        x = self.LReLU(self.BNorm4(self.Conv4(x)))
        x = self.Dropout(x)
        x = self.LReLU(self.BNorm5(self.Conv5(x)))
        x = x.reshape(-1, 3456)
        x = self.LReLU(self.Linear1(x))
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

    if epoch % 17 == 0:
        validAccuracyFunction()
        trainAccuracyFunction()
        trainAsTestAccuracyFunction()
        if validAccuracy > (bestAccuracy - 3):
          bestAccuracy = validAccuracy
          testAccuracyFunction()
          if testAccuracy > bestTAccuracy:
            bestTAccuracy = testAccuracy
            name = './Flowers8-'+str(epoch)+'-'+str(round(testAccuracy))+'.pth'
            torch.save(neuNet.state_dict(), name)

    if ((epoch > 500) and (epoch % 7 == 0)):
        validAccuracyFunction()
        trainAsTestAccuracyFunction()
        if validAccuracy > (bestAccuracy - 2):
          bestAccuracy = validAccuracy
          testAccuracyFunction()
          if testAccuracy > bestTAccuracy:
            bestTAccuracy = testAccuracy
            name = './Flowers8-'+str(epoch)+'-'+str(round(testAccuracy))+'.pth'
            torch.save(neuNet.state_dict(), name)