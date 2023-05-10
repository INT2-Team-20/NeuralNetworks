import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler

#Consisted dropouts with larger dropout in between linear layers

epochs = 1000

trainingTransform = transforms.Compose(
    [
     transforms.RandomResizedCrop((224,224),(0.5, 1.0)),
     transforms.RandomRotation(20),
     transforms.RandomHorizontalFlip(0.5),
     transforms.RandomVerticalFlip(0.3),
     transforms.ColorJitter(0.3, 0.025),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testTransform = transforms.Compose(
    [
     
     transforms.CenterCrop(448), #started resizing after center crop
     transforms.Resize(224),
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
        self.Linear1 = nn.Linear(2048, 1024) 
        self.Linear2 = nn.Linear(1024, 102)
        self.Dropout = nn.Dropout(0.125)
        self.Dropout2 = nn.Dropout(0.3)
        self.Conv1 = nn.Conv2d(3, 64, 5) 
        self.BNorm1 = nn.BatchNorm2d(64)
        self.Conv2 = nn.Conv2d(64, 128, 5)
        self.BNorm2 = nn.BatchNorm2d(128) #added multiple layers following inspiration form popular machine learning models
        self.Conv3 = nn.Conv2d(128, 256, 5,2)
        self.BNorm3 = nn.BatchNorm2d(256)
        self.Conv4= nn.Conv2d(256,512,5,2)
        self.BNorm4 = nn.BatchNorm2d(512)
        self.Pool = nn.MaxPool2d((2,2),(2,2))
        self.LReLU = nn.LeakyReLU()
        self.PReLU=nn.PReLU()

    def forward(self, x):
        #print(inputs.shape)
        x = self.PReLU(self.BNorm1(self.Conv1(x))) #110
        x = self.Dropout(x)
        x = self.Pool(x) #55
        #print("first pool later "+str(x.size()))

      
        x = self.PReLU(self.BNorm2(self.Conv2(x))) 
        x = self.Dropout(x)
        x = self.Pool(x) #27
       # print("second pool later "+str(x.size()))

        x = self.PReLU(self.BNorm3(self.Conv3(x))) 
        x = self.Dropout(x)
        x = self.Pool(x)#13
        #print("third pool later "+str(x.size()))

        x = self.PReLU(self.BNorm4(self.Conv4(x))) 
        x = self.Dropout(x)
        x = self.Pool(x)# (5/5)
        #print("after fourth pool  "+str(x.size()))

        # Flatten the tensor to a 1D tensor
        x = torch.flatten(x, start_dim=1)

        # Reshape the tensor to size [24, 2048]
       # print(x.shape)
        #print("resieze"+ str(x))
        x = self.PReLU(self.Linear1(x))
        x = self.Dropout2(x)
        x = self.Linear2(x)
       # print(x.shape)
        return x

neuNet = NeuralN().to("cuda")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(neuNet.parameters(),  0.00005 , weight_decay=0.01) #previous rate:0.00005  , 0.001
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.8)
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

    scheduler.step()
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
