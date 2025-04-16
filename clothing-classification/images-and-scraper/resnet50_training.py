import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms, models
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import os
from tqdm import tqdm


# Define constants
BATCH_SIZE = 32
NUM_EPOCHS = 13
NUM_FOLDS = 5
LEARNING_RATE = 0.005
DROPOUT_RATE = 0.4
BS_SAMPLES = 50


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


class ClothingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Define class names in lowercase for case-insensitive matching
        self.classes_lower = {
            'ayakkabı': 'Ayakkabı',
            'etek': 'Etek',
            'gömlek': 'Gömlek',
            'kazak': 'Kazak',
            'pantolon': 'Pantolon',
            'tişört': 'Tişört',
            'çanta': 'Çanta',
            'şapka': 'Şapka',
            'şort': 'Şort'
        }

        self.classes = list(self.classes_lower.values())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.images = []
        self.labels = []

        print(f"Looking for images in: {root_dir}")
        print(f"Available classes: {self.classes}")

        # Walk through directory structure and print what we find
        for root, dirs, files in os.walk(root_dir):
            print(f"\nChecking directory: {root}")

            # Check if current directory name (lowercase) matches any class
            current_dir = os.path.basename(root).lower()
            if current_dir in self.classes_lower:
                class_name = self.classes_lower[current_dir]
                valid_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                print(f"Found {len(valid_files)} valid images for class {class_name} in {root}")

                for file in valid_files:
                    self.images.append(os.path.join(root, file))
                    self.labels.append(self.class_to_idx[class_name])

        print(f"\nTotal images found: {len(self.images)}")
        if len(self.images) > 0:
            print(f"Distribution of classes:")
            unique_labels, counts = np.unique(self.labels, return_counts=True)
            for label, count in zip(unique_labels, counts):
                print(f"{self.classes[label]}: {count} images")
        else:
            print("No images found! Please check your directory structure.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]

            if self.transform:
                image = self.transform(image)

            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            raise e


def create_model(dropout_rate=DROPOUT_RATE):
    model = models.resnet50(pretrained=True)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer with Dropout and a new FC layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_rate),  # Add dropout before the final layer
        nn.Linear(num_features, 9)  # 9 classes
    )

    return model.to(DEVICE)




def train_model(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(train_loader), 100. * correct / total


def evaluate_model(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Evaluating"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / len(val_loader), 100. * correct / total, all_preds, all_labels


def plot_confusion_matrix(true_labels, predictions, classes):
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def bootstrap_evaluate(model, dataset, num_samples=BS_SAMPLES):
    results = []
    n_samples = len(dataset)

    for _ in tqdm(range(num_samples), desc="Bootstrap evaluation"):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        sampler = SubsetRandomSampler(indices)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)

        _, accuracy, _, _ = evaluate_model(model, loader, criterion)
        results.append(accuracy)

    return np.mean(results), np.std(results)


def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(15, 5))

    # Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()


# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Create dataset with debug information
print("\nInitializing dataset...")
dataset = ClothingDataset(root_dir='clothing_images', transform=transform)

if len(dataset) == 0:
    print("\nNo images were found! Please check:")
    print("1. The path to your images directory is correct")
    print("2. The directory structure matches expected format")
    print("3. The folder names match the class names (case-insensitive)")
    print("4. There are valid image files in the folders")
    exit()

# K-fold Cross Validation
kfold = KFold(n_splits=NUM_FOLDS, shuffle=True)
fold_results = []

for fold, (train_ids, val_ids) in enumerate(kfold.split(range(len(dataset)))):
    print(f'\nFOLD {fold + 1}/{NUM_FOLDS}')
    print('--------------------------------')

    train_sampler = SubsetRandomSampler(train_ids)
    val_sampler = SubsetRandomSampler(val_ids)

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

    model = create_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer)
        val_loss, val_acc, predictions, true_labels = evaluate_model(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f'Epoch {epoch + 1}/{NUM_EPOCHS}')
        print(f'Training Loss: {train_loss:.4f}, Training Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_model_fold_{fold}.pth')

    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)

    # Load best model for final evaluation
    model.load_state_dict(torch.load(f'best_model_fold_{fold}.pth'))
    _, final_acc, final_preds, final_labels = evaluate_model(model, val_loader, criterion)
    fold_results.append(final_acc)

    # Plot confusion matrix for this fold
    plot_confusion_matrix(final_labels, final_preds, dataset.classes)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(final_labels, final_preds,
                                target_names=dataset.classes))

    # Bootstrap evaluation for this fold
    bootstrap_mean, bootstrap_std = bootstrap_evaluate(model, dataset)
    print(f'Bootstrap Results - Mean Accuracy: {bootstrap_mean:.2f}%, Std: {bootstrap_std:.2f}%')

# Print overall k-fold results
print('\nK-fold Cross Validation Results')
print('--------------------------------')
print(f'Average Accuracy: {np.mean(fold_results):.2f}%')
print(f'Std Deviation: {np.std(fold_results):.2f}%')