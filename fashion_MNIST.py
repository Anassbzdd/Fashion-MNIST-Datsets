import torch 
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt 

# 1. Download the Data
print('Dowloading Fashion MNIST ...')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])

train_dataset = datasets.FashionMNIST(
    root = './data',
    download = True,
    train = True,
    transform = transform
)

test_dataset = datasets.FashionMNIST(
    root = './data',
    download = True,
    train = False,
    transform = transform
)

# 2. DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=32, 
    shuffle = True
)

test_loader = DataLoader(
    test_dataset,
    batch_size= 32, 
    shuffle = False
)

print(f'Training samples : {len(train_dataset)}')
print(f'Testing samples : {len(test_dataset)}')

# The Model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.inp = nn.Linear(784, 128)
        self.hidden = nn.Linear(128, 64)
        self.out = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self , x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.inp(x))
        x = self.relu(self.hidden(x))
        x = self.out(x)
        return x

# Initialize model, loss, and optimizer
model = MyModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters() , lr = 0.001)

print('Model Architecture:')
print(model)

def train(model , loader , criterion , optimizer , device):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for batch_idx , (images , target) in enumerate(loader):
        images , target = images.to(device), target.to(device)

        # Zero Gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs , target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _ , predicted = torch.max(outputs.data , 1)
        total += target.size(0)
        correct += (predicted == target ).sum().item()
        running_loss += loss.item()

        if (batch_idx + 1) % 200 == 0:
            print(f'Batch [{batch_idx + 1}/{len(loader)}],'
                  f'Loss : {loss.item():.4f},'
                  f'Accuracy : {100 * correct / total:.2f}')
            
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total
    return epoch_loss , epoch_acc
    
# Validation Function
def validate(model , loader , criterion,  device):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images , labels in loader :
            images , labels = images.to(device), labels.to(device)

            # Forward pass Only
            outputs = model(images)
            loss = criterion(outputs , labels)

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

        loss_val = running_loss / len(loader)
        acc_val = 100 * correct / total
        return loss_val , acc_val
    
# Main Training Loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f'\n Using device : {device}')

num_epochs = 10
train_losses , train_accs = [], []
val_losses , val_accs = [], []

print('Starting Training ...')
print('__' *50)

for epoch in range(num_epochs):
    print(f'\n Epoch [{epoch + 1} / {num_epochs}]')
    print('__' * 50)

    # Test 
    loss_train , acc_train = train(model, train_loader , criterion , optimizer, device)
    train_losses.append(loss_train)
    train_accs.append(acc_train)

    # Validation
    loss_val , acc_val = validate(model, test_loader , criterion , device)
    val_losses.append(loss_val)
    val_accs.append(acc_val)

    print(f"\n  Training   - Loss: {loss_train:.4f}, Accuracy: {acc_train:.2f}%")
    print(f"\n  Validation   - Loss: {loss_val:.4f}, Accuracy: {acc_val:.2f}%")
    print("=" * 60)

print("\n✅ Training Complete!")
print(f"Final Test Accuracy: {val_accs[-1]:.2f}%")

# Plot Results 
plt.figure(figsize=(10, 8))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(train_losses, label = 'Train Loss')
plt.plot(val_losses, label = 'Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# Accuracy Plot
plt.subplot(1, 2, 2)
plt.plot(train_accs, label = 'Train Acc')
plt.plot(val_accs, label = 'Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy %')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('results.png')
plt.show()
