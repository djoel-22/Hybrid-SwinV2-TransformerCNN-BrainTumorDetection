import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
from timm import create_model
import torch.nn.functional as F

class HybridSwinCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(HybridSwinCNN, self).__init__()
        self.swin = create_model('swinv2_large_window12_192_22k', pretrained=False, num_classes=0)
        self.cnn_branch = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((6, 6))
        )
        
        # Calculate combined features dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 192, 192)
            swin_feat = self.swin(dummy).shape[1]
            cnn_feat = self.cnn_branch(dummy).view(1, -1).shape[1]
            total_features = swin_feat + cnn_feat
            
        self.fc = nn.Sequential(
            nn.Linear(self.swin.num_features + 128 * 6 * 6, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        swin_feat = self.swin(x)
        cnn_feat = self.cnn_branch(x)
        cnn_feat = cnn_feat.view(cnn_feat.size(0), -1)
        combined = torch.cat((swin_feat, cnn_feat), dim=1)
        return self.fc(combined)

# ... (REST OF YOUR CODE REMAINS EXACTLY THE SAME FROM train() TO THE END) ...

def train(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct = 0, 0
    loop = tqdm(loader, total=len(loader), desc="Training", leave=False)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        loop.set_postfix(loss=loss.item(), acc=100. * correct / ((loop.n + 1) * loader.batch_size))
    return correct / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, total=len(loader), desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
    return correct / len(loader.dataset)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dir = 'dataset/Training'
    test_dir = 'dataset/Testing'

    transform_train = transforms.Compose([
        transforms.Resize((192, 192)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(10, scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    transform_test = transforms.Compose([
        transforms.Resize((192, 192)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform_test)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)

    print("Dataset Summary:")
    print(f"Training Images: {len(train_dataset)}")
    print(f"Testing Images : {len(test_dataset)}")
    print(f"Classes        : {train_dataset.classes}")

    model = HybridSwinCNN(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    num_epochs = 20
    os.makedirs("model", exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch [{epoch}/{num_epochs}]")
        train_acc = train(model, train_loader, optimizer, criterion, device)
        print(f"Train Accuracy: {train_acc * 100:.2f}%")

        test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Test Accuracy : {test_acc * 100:.2f}%")

        model_path = f"model/model_epoch_{epoch}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved: {model_path}")

if __name__ == '__main__':
    main()
