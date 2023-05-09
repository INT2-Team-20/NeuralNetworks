import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

epochs = 100

trainingTransform = transforms.Compose(
    [transforms.Resize(448),
     transforms.RandomResizedCrop(224,(0.5, 1.0)),
     transforms.CenterCrop(224),
     transforms.RandomRotation(20),
     transforms.RandomHorizontalFlip(0.4),
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

trainingDataloader = torch.utils.data.DataLoader(trainingData, 24, True)
testDataloader = torch.utils.data.DataLoader(testData, 24, True)

class NeuralN(nn.Module):
    def __init__(self):
        super().__init__()
        self.Linear1 = nn.Linear(150528, 2048)
        self.Linear2 = nn.Linear(2048, 1024)
        self.Linear3 = nn.Linear(1024, 102)

    def forward(self, x):
        x = F.relu(self.Linear1(x))
        x = F.relu(self.Linear2(x))
        x = self.Linear3(x)
        return x

neuNet = NeuralN().to("cuda")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(neuNet.parameters(), 0.00005)

for epoch in range(epochs):
    print(epoch)
    for i, (inputs,labels) in enumerate(trainingDataloader):
        inputs = inputs.reshape(-1, 150528).to("cuda")
        labels = labels.to("cuda")

        optimizer.zero_grad()

        outputs = neuNet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

print('Finished Training')
torch.save(neuNet.state_dict(), './Flowers3.pth')