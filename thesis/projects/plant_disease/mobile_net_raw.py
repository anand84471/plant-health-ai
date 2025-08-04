import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from thesis.utils.pytorch import train

from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn import Module
from torch import device
import time
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision


def loop_fn(mode: str, dataset: Dataset, dataloader: DataLoader, 
                model: Module, criterion: Module , optimizer, device: device):
    """
    Args:
        mode (str): 'train' or 'val'
        dataset (torch.utils.data.Dataset): Dataset
        dataloader (torch.utils.data.DataLoader): Dataloader
        model (torch.nn.Module): Model
        criterion (torch.nn.Module): Criterion
        optimizer (torch.optim.Optimizer): Optimizer
        device (torch.device): Device
    """
    if mode == 'train':
        model.train()
    elif mode == 'val':
        model.eval()
    
    cost = correct = 0

    for feature, target in tqdm(dataloader, desc=mode.title()):
        feature, target = feature.to(device), target.to(device)
        output = model(feature)
        loss = criterion(output, target)

        if mode == 'train':
            loss.backward() # compute gradient
            optimizer.step() # update weights
            optimizer.zero_grad() # reset gradient
        
        cost += loss.item() * feature.shape[0] # sum of loss
        correct += (output.argmax(1) == target).sum().item() # sum of correct predictions
    cost = cost/len(dataset) # average loss
    acc = correct/len(dataset) # accuracy
    return cost, acc
    


def evaluate(model: Module, dataloader: DataLoader , device: device):
    """
    Args:
        model (torch.nn.Module): Model
        dataloader (torch.utils.data.DataLoader): Dataloader
        device (torch.device): Device
    """

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"✅ Accuracy: {acc:.4f} ({correct}/{total})")
    return acc
    

def train(model_name: str, epoch: int, model: Module, train_data: Dataset, train_loader: DataLoader, test_data: Dataset, test_loader: DataLoader, criterion: Module, optimizer, device: device):
    """
    Args:
        model_name (str): Model name
        epoch (int): Number of epochs
        model (torch.nn.Module): Model
        train_data (torch.utils.data.Dataset): Training dataset
        train_loader (torch.utils.data.DataLoader): Training dataloader
        test_data (torch.utils.data.Dataset): Testing dataset
        test_loader (torch.utils.data.DataLoader): Testing dataloader
        criterion (torch.nn.Module): Criterion
        optimizer (torch.optim.Optimizer): Optimizer
        device (torch.device): Device
    """
    import torch
    best_weight = None
    best_train = [0, 0, 0, 0, 0]
    for i in range(epoch):
        print("="*30, f"Epoch {i+1}:")
        train_cost, train_score = loop_fn('train', train_data, train_loader, model, criterion, optimizer, device)
        print(f"===Train:\t|\tAccuracy: {train_score:.4f}\t|\tLoss: {train_cost:.4f}")
        with torch.no_grad():
            test_cost, test_score = loop_fn('val', test_data, test_loader, model, criterion, optimizer, device)
            print(f"===Valid:\t|\tAccuracy: {test_score:.4f}\t|\tLoss: {test_cost:.4f}")


        if best_train[0] <= train_score:
            best_train = list([train_score, train_cost, test_score, test_cost, i+1])
            best_weight = model.state_dict()
        print("==="*30, "\n\n\n")

    print(f"Best checkpoinpt: {best_train[4]}\nTrain Accuracy: {best_train[0]}\t|\tTrain Loss: {best_train[1]}\nTest Accuracy: {best_train[2]}\t|\tTest Loss: {best_train[3]}")
    evaluate(model, test_loader, device)

    id_cp = best_train[4]
    torch.save(best_weight, f"model_{model_name}_plantvillage_{id_cp}.pth")


def train_with_plots(model_name: str, epoch: int, model: Module, train_data: Dataset, train_loader: DataLoader,
          test_data: Dataset, test_loader: DataLoader, criterion: Module, optimizer, device: device):
    """
    Train and evaluate the model.

    Args:
        model_name (str): Name of the model for saving checkpoint.
        epoch (int): Number of epochs.
        model (torch.nn.Module): Model to train.
        train_data (torch.utils.data.Dataset): Training dataset.
        train_loader (torch.utils.data.DataLoader): Training dataloader.
        test_data (torch.utils.data.Dataset): Testing dataset.
        test_loader (torch.utils.data.DataLoader): Testing dataloader.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Computation device (CPU or CUDA).
    """

    train_accs, train_losses = [], []
    val_accs, val_losses = [], []
    times = []

    best_weight = None
    best_train = [0, 0, 0, 0, 0]

    for i in range(epoch):
        print("="*30, f"Epoch {i + 1}/{epoch}")

        start_time = time.time()

        # Training phase
        train_cost, train_score = loop_fn('train', train_data, train_loader, model, criterion, optimizer, device)
        train_accs.append(train_score)
        train_losses.append(train_cost)

        # Validation phase
        with torch.no_grad():
            test_cost, test_score = loop_fn('val', test_data, test_loader, model, criterion, optimizer, device)
        val_accs.append(test_score)
        val_losses.append(test_cost)

        epoch_time = time.time() - start_time
        times.append(epoch_time)

        print(f"Train:\tAccuracy: {train_score:.4f}\tLoss: {train_cost:.4f}")
        print(f"Valid:\tAccuracy: {test_score:.4f}\tLoss: {test_cost:.4f}")
        print(f"Time taken: {epoch_time:.2f} seconds")
        
        # Save best checkpoint (based on training accuracy)
        if best_train[0] <= train_score:
            best_train = [train_score, train_cost, test_score, test_cost, i + 1]
            best_weight = model.state_dict()

        print("="*90, "\n")

    # Save the best model
    checkpoint_path = f"model_{model_name}_plantvillage_epoch{best_train[4]}.pth"
    torch.save(best_weight, checkpoint_path)

    print(f"Best Epoch: {best_train[4]}")
    print(f"Train Accuracy: {best_train[0]:.4f} | Train Loss: {best_train[1]:.4f}")
    print(f"Test Accuracy: {best_train[2]:.4f}  | Test Loss: {best_train[3]:.4f}")
    print(f"Model saved to: {checkpoint_path}")

    # Final evaluation
    model.load_state_dict(best_weight)
    evaluate(model, test_loader, device)

    # Plot metrics
    _plot_training_curves(train_accs, val_accs, train_losses, val_losses)


def _plot_training_curves(train_accs, val_accs, train_losses, val_losses):
    """Utility to plot accuracy and loss curves"""
    epochs = range(1, len(train_accs) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accs, label='Train Acc')
    plt.plot(epochs, val_accs, label='Val Acc')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Epochs")
    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()

    plt.tight_layout()
    plt.show()




# Define data transforms
# MobileNetV2 expects 224x224 input images and specific normalization
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),          # phóng to rồi crop ngẫu nhiên vùng 224x224
        transforms.RandomHorizontalFlip(p=0.5),     # lật ngang ảnh ngẫu nhiên
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # thay đổi màu sắc
        transforms.RandomRotation(degrees=15),      # xoay ngẫu nhiên ±15 độ
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # dịch ảnh
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5), # hiệu ứng phối cảnh
        transforms.ToTensor(),                      # chuyển thành tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([ # Often same as validation for consistent evaluation
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# base_dir = "/Users/anand/Desktop/1mg/repos/thesis/thesis/data/PlantDoc-Dataset"
base_dir = "/Users/anand/Desktop/temp/Data for Identification of Plant Leaf Diseases Using a 9-layer Deep Convolutional Neural Network/Plant_leave_diseases_dataset_without_augmentation"
base_dataset = datasets.ImageFolder(base_dir)

total_size = len(base_dataset)
print('Total size:', total_size)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

print('Train size:', train_size)
print('Val size:', val_size)
print('Test size:', test_size)

train_data, val_data, test_data = random_split(
    base_dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

train_data.dataset.transform = data_transforms["train"]
val_data.dataset.transform = data_transforms["val"]
test_data.dataset.transform = data_transforms["test"]

batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_data, shuffle=True, pin_memory=True)



# device = torch.device("mps" if torch.mps.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
print("Using device:", device)



model = torchvision.models.mobilenet_v3_large(weights="DEFAULT")

# Get number of classes
num_classes = len(base_dataset.classes) 
print("Num classes: ", num_classes)

# Check classifier structure and get input features
print("Classifier:", model.classifier)

# For MobileNetV3, the final layer is usually at index -1 or index 3
# Let's try the last layer
final_layer = model.classifier[-1]  # or try model.classifier[3]
in_features = final_layer.in_features
print("No of features: ", in_features)

# Replace the final layer
model.classifier[-1] = nn.Linear(in_features, num_classes)

# Move to device
model = model.to(device)

# Freeze feature extraction layers
for param in model.features.parameters():
    param.requires_grad = False

# Setup loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



num_epochs = 30
print("device", device)
train("mobile_net_v3", num_epochs, model, train_data, train_loader, test_data, test_loader, criterion, optimizer, device)