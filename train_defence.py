from attacks.fgsm import fgsm_attack
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DATA
transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# MODEL
model = models.resnet18(weights="DEFAULT")
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# TRAIN WITH DEFENSE
epochs = 2
for epoch in range(epochs):
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        images.requires_grad = True

        outputs = model(images)
        loss = criterion(outputs, labels)

        # FGSM gradient
        data_grad = torch.autograd.grad(loss, images, retain_graph=True)[0]

        adv_images = fgsm_attack(images, 0.1, data_grad)

        outputs_adv = model(adv_images)
        loss_adv = criterion(outputs_adv, labels)

        total_loss = loss + loss_adv

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

# SAVE
torch.save(model.state_dict(), "models/defense_model.pth")
print("Defense model saved!")