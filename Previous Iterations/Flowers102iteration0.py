import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

epochs = 1000

transform = transforms.Compose(
    [transforms.CenterCrop(128),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainingData = torchvision.datasets.Flowers102(
    root="data",
    split="train",
    download=True,
    transform=transform,
)

testData = torchvision.datasets.Flowers102(
    root="data",
    split="test",
    download=True,
    transform=transform,
)

trainingDataloader = torch.utils.data.DataLoader(trainingData, 12, True)
testDataloader = torch.utils.data.DataLoader(testData, 12, True)

class NeuralN(nn.Module):
    def __init__(self):
        super().__init__()
        self.Linear1 = nn.Linear(49152, 434)
        self.Linear2 = nn.Linear(434, 163)
        self.Linear3 = nn.Linear(163, 102)

    def forward(self, x):
        x = F.relu(self.Linear1(x))
        x = F.relu(self.Linear2(x))
        x = self.Linear3(x)
        return x

neuNet = NeuralN().to("cuda")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(neuNet.parameters(), 0.001)

for epoch in range(epochs):
    print(epoch)
    for i, (inputs,labels) in enumerate(trainingDataloader):
        inputs = inputs.reshape(-1, 49152).to("cuda")
        labels = labels.to("cuda")

        optimizer.zero_grad()

        outputs = neuNet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

print("Finished. 'Saving to Flowers.pth'")
torch.save(neuNet.state_dict(), './Flowers.pth')