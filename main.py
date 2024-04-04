import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import numpy as np

# Set up TensorBoard
writer = SummaryWriter()

class USPSMLP(nn.Module):
    def __init__(self):
        super(USPSMLP, self).__init__()
        self.fc1 = nn.Linear(256, 528)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(528, 125)
        self.fc3 = nn.Linear(125, 10)


    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)

        return out

class USPSCNN(nn.Module):
    def __init__(self):
        super(USPSCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 18, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(18, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(32 * 4 * 4, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images
            labels = labels
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:

                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
                writer.add_scalar('training_loss', loss.item(), epoch*total_step + i)
    print()

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images
            labels = labels
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    recall = recall_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)
    # Add precision-recall curve for MLP
    writer.add_pr_curve('MLP', torch.tensor(all_labels), torch.tensor(all_preds))

    # Add loss function for MLP
    return accuracy, recall, precision, cm

from torchvision import datasets, transforms


transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize to range [-1, 1]
])

# Load USPS training dataset
train_dataset = datasets.USPS(root='./data', train=True, download=True, transform=transform)

# Load USPS testing dataset
test_dataset = datasets.USPS(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=50, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=50, shuffle=False)

# Initialize models
mlp_model = USPSMLP()
cnn_model = USPSCNN()

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

# Train MLP
train_model(mlp_model, train_loader, criterion, mlp_optimizer, num_epochs=5)
# Train CNN
train_model(cnn_model, train_loader, criterion, cnn_optimizer, num_epochs=5)

# Evaluate MLP
mlp_accuracy, mlp_recall, mlp_precision, mlp_cm = evaluate_model(mlp_model, test_loader)
# Evaluate CNN
cnn_accuracy, cnn_recall, cnn_precision, cnn_cm = evaluate_model(cnn_model, test_loader)

print()
print("MLP Accuracy:", mlp_accuracy)
print()
print("MLP Recall:", mlp_recall)
print()
print("MLP Precision:", mlp_precision)
print()
print("MLP Confusion Matrix:")
print()
print(mlp_cm)
print()
print()
print()

print("CNN Accuracy:", cnn_accuracy)
print()

print("CNN Recall:", cnn_recall)
print()

print("CNN Precision:", cnn_precision)
print()

print("CNN Confusion Matrix:")
print()

print(cnn_cm)
print()

