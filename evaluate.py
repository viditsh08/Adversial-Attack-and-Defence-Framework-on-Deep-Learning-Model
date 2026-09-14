from attacks.fgsm import fgsm_attack
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DATA
transform = transforms.Compose([transforms.ToTensor()])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

# LOAD MODEL FUNCTION
def load_model(path):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(path))
    model = model.to(device)
    model.eval()
    return model

baseline_model = load_model("models/baseline_model.pth")
defense_model = load_model("models/defense_model.pth")

# TEST FUNCTION
def test_model(model, epsilon):
    correct = 0
    total = 0

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        images.requires_grad = True

        outputs = model(images)
        _, init_pred = torch.max(outputs, 1)

        if init_pred.item() != labels.item():
            continue

        loss = nn.CrossEntropyLoss()(outputs, labels)

        model.zero_grad()
        loss.backward()

        data_grad = images.grad.data
        perturbed_data = fgsm_attack(images, epsilon, data_grad)

        outputs = model(perturbed_data)
        _, final_pred = torch.max(outputs, 1)

        if final_pred.item() == labels.item():
            correct += 1

        total += 1

    return correct / total

# FINAL COMPARISON
epsilon = 0.1

baseline_acc = test_model(baseline_model, epsilon)
defense_acc = test_model(defense_model, epsilon)

print("\n===== FINAL RESULT =====")
print(f"Baseline Model Accuracy under attack: {baseline_acc*100:.2f}%")
print(f"Defense Model Accuracy under attack:  {defense_acc*100:.2f}%")


#  ADD VISUALIZATION HERE

import matplotlib.pyplot as plt

examples = []

for images, labels in testloader:
    images, labels = images.to(device), labels.to(device)
    images.requires_grad = True

    outputs = defense_model(images)
    _, init_pred = torch.max(outputs, 1)

    loss = nn.CrossEntropyLoss()(outputs, labels)
    defense_model.zero_grad()
    loss.backward()

    data_grad = images.grad.data
    perturbed_data = fgsm_attack(images, 0.1, data_grad)

    outputs = defense_model(perturbed_data)
    _, final_pred = torch.max(outputs, 1)

    if len(examples) < 5:
        examples.append((
            images.squeeze().detach().cpu(),
            perturbed_data.squeeze().detach().cpu(),
            init_pred.item(),
            final_pred.item()
        ))

# SHOW IMAGES
plt.figure(figsize=(10,5))

for i, (orig, adv, pred, adv_pred) in enumerate(examples):
    plt.subplot(2, 5, i+1)
    plt.imshow(orig.permute(1,2,0))
    plt.title(f"Orig: {pred}")
    plt.axis('off')

    plt.subplot(2, 5, i+6)
    plt.imshow(adv.permute(1,2,0))
    plt.title(f"Adv: {adv_pred}")
    plt.axis('off')

plt.show()
