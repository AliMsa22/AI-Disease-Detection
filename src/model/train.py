import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import csv
from torchvision.models import resnet34, ResNet34_Weights, efficientnet_b0, EfficientNet_B0_Weights, densenet121, DenseNet121_Weights, resnet50, ResNet50_Weights
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
from collections import Counter
from torch.amp import autocast, GradScaler
import numpy as np

from src.model.chestxray_cnn import ChestXrayCNN
from src.model.chestxray_cnn_v2 import ChestXrayCNNv2
from src.model.pneumonia_cnn import PneumoniaCNN
# from pneumonia_cnn import PneumoniaCNNv2
# from chestxray_cnn import ChestXrayCNN
# from chestxray_cnn_v2 import ChestXrayCNNv2
from dataset.chestxray_dataset import ChestXrayDataset




# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2, alpha=None, reduction='mean'):
#         """
#         Focal Loss for multi-class classification.
#         Args:
#             gamma (float): Focusing parameter. Default is 2.
#             alpha (Tensor or None): Class weights. Default is None.
#             reduction (str): Reduction method ('none', 'mean', 'sum'). Default is 'mean'.
#         """
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.reduction = reduction

#     def forward(self, inputs, targets):
#         """
#         Args:
#             inputs (Tensor): Predicted logits (before softmax) of shape (batch_size, num_classes).
#             targets (Tensor): Ground truth labels of shape (batch_size).
#         Returns:
#             Tensor: Computed focal loss.
#         """
#         # Convert logits to probabilities
#         probs = F.softmax(inputs, dim=1)
        
#         # Get the probabilities of the true class
#         targets_one_hot = F.one_hot(targets, num_classes=probs.size(1)).float()
#         p_t = (probs * targets_one_hot).sum(dim=1)  # Shape: (batch_size,)

#         # Compute the focal loss
#         focal_weight = (1 - p_t) ** self.gamma
#         log_p_t = torch.log(p_t + 1e-8)  # Add epsilon to avoid log(0)
#         loss = -focal_weight * log_p_t

#         # Apply class weights (if provided)
#         if self.alpha is not None:
#             alpha_t = (self.alpha * targets_one_hot).sum(dim=1)  # Shape: (batch_size,)
#             loss = alpha_t * loss

#         # Apply reduction
#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         else:  # 'none'
#             return loss


def get_model(arch, num_classes):
    if arch == "v1":
        return ChestXrayCNN(num_classes=num_classes)

    elif arch == "v2":
        return ChestXrayCNNv2(num_classes=num_classes)
    
    elif arch == "pneumonia":
        return PneumoniaCNN(num_classes=num_classes)

    elif arch == "resnet34":
        model = resnet34(weights=ResNet34_Weights.DEFAULT)

        # Modify the first convolutional layer to accept 1-channel input
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        nn.init.kaiming_normal_(model.conv1.weight, nonlinearity='relu')

        # Modify the final layer for num_classes outputs
        model.fc = nn.Sequential(
            nn.Dropout(0.2),  # Adjust dropout rate if necessary
            nn.Linear(model.fc.in_features, num_classes)
        )
        nn.init.xavier_uniform_(model.fc[1].weight)
        nn.init.zeros_(model.fc[1].bias)
        return model
    
    elif arch == "resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Modify the first convolutional layer to accept 1-channel input
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(model.conv1.weight, nonlinearity='relu')

        # Modify the final layer for num_classes outputs
        model.fc = nn.Sequential(
            nn.Dropout(0.2),  # Adjust dropout rate if necessary
            nn.Linear(model.fc.in_features, num_classes)
        )
        nn.init.xavier_uniform_(model.fc[1].weight)
        nn.init.zeros_(model.fc[1].bias)
        return model

    elif arch == "efficientnet":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        # Modify the first convolutional layer to accept 1-channel input
        model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        nn.init.kaiming_normal_(model.features[0][0].weight, nonlinearity='relu')

        # Modify the final layer for num_classes outputs
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        nn.init.xavier_uniform_(model.classifier[1].weight)
        nn.init.zeros_(model.classifier[1].bias)
        return model

    elif arch == "densenet":
        model = densenet121(weights=DenseNet121_Weights.DEFAULT)

        # Modify the first convolutional layer to accept 1-channel input
        model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(model.features.conv0.weight, nonlinearity='relu')
        
        # Modify the final layer for num_classes outputs
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        nn.init.xavier_uniform_(model.classifier.weight)
        nn.init.zeros_(model.classifier.bias)
        return model

    else:
        raise ValueError("Unsupported architecture. Use 'v1', 'v2', 'resnet', 'efficientnet', or 'densenet'.")


# Warm-up function
def adjust_learning_rate(optimizer, epoch, warmup_epochs=5, base_lr=0.001):
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs  # Gradually increase LR
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f"Warm-up learning rate: {lr}")


def main(args):
    # Set the device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the full dataset
    train_dataset = ChestXrayDataset(
        csv_file='data/train.csv',
        use_preprocessed=args.use_preprocessed,
        train=True
    )

    val_dataset = ChestXrayDataset(
        csv_file='data/val.csv',
        use_preprocessed=args.use_preprocessed,
        train=False
    )

    # Calculate class weights
    label_counts = Counter(train_dataset.data['Finding Labels'])  # Assuming 'Finding Labels' contains class labels

    class_names = ['Effusion', 'Mass', 'Nodule', 'No Finding']  # Actual class labels
    classes = np.array(class_names)  # Convert to numpy array
    print("Labels in dataset:", label_counts.keys())
    print("Expected classes:", classes)

    # Compute class weights using actual class labels
    class_weights = compute_class_weight('balanced', classes=classes, y=list(label_counts.elements()))
    class_weights_tensor = torch.tensor(class_weights).float().to(device)
    print("Class Weights:", class_weights)

    # Modify DataLoader to shuffle the dataset
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,  # Shuffle the dataset
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    # Use class weights in the loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Initialize the model based on the specified architecture
    model = get_model(args.arch, num_classes=4).to(device)

    # Freeze pretrained layers for the first 5 epochs (if using a pretrained model)
    if args.arch in ["resnet34", "resnet50", "efficientnet", "densenet"]:
        att_name = {
            "resnet34": "fc",
            "resnet50": "fc",
            "efficientnet": "classifier",
            "densenet": "classifier"
        }[args.arch]

        head = getattr(model, att_name)
        # freeze all
        for p in model.parameters():
            p.requires_grad = False
        # unfreeze just the head
        for p in head.parameters():
            p.requires_grad = True

    # Define the optimizer
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=1e-4)

    # Use ReduceLROnPlateau as the learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-6)

    scaler = GradScaler()

    # Define the directory for saving metrics
    metrics_dir = "src/model/training_metrics"
    os.makedirs(metrics_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Define the full path for the CSV file
    csv_file_path = os.path.join(metrics_dir, f"metrics_{args.arch}_{'pre' if args.use_preprocessed else 'orig'}.csv")

    # Early stopping parameters
    best_val_acc = 0
    patience = 10
    epochs_no_improve = 0
    best_model_wts = None

    # Open a CSV file to save metrics
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Epoch", "Train Loss", "Train Accuracy", "Val Loss", "Val Accuracy"])

        for epoch in range(args.epochs):
            # Apply warm-up for the first 5 epochs
            adjust_learning_rate(optimizer, epoch)

            # Unfreeze all layers after the warm-up phase
            if epoch == 5 and args.arch in ["resnet34", "resnet50", "efficientnet", "densenet"]:
                for p in model.parameters():
                    p.requires_grad = True
                optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-6)
                print("Unfroze all layers and reset optimizer + scheduler for fine-tuning.")

            # Training phase
            model.train()
            running_loss = 0.0
            all_labels = []
            all_preds = []

            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with autocast(device_type='cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

            train_accuracy = accuracy_score(all_labels, all_preds) * 100
            train_loss = running_loss / len(train_loader)

            # Validation phase
            model.eval()
            val_loss = 0.0
            all_labels = []
            all_preds = []

            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc=f"Validation {epoch+1}/{args.epochs}"):
                    inputs, labels = inputs.to(device), labels.to(device)

                    with autocast(device_type='cuda'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(predicted.cpu().numpy())

            val_accuracy = accuracy_score(all_labels, all_preds) * 100
            val_loss = val_loss / len(val_loader)

            # Print and save metrics
            print(f"Epoch [{epoch+1}/{args.epochs}], "
                  f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

            csv_writer.writerow([epoch + 1, train_loss, train_accuracy, val_loss, val_accuracy])

            # Update learning rate scheduler
            scheduler.step(val_loss)  # Use validation loss to adjust learning rate
            print(f"Learning rate updated to: {optimizer.param_groups[0]['lr']}")

            # Early stopping
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                best_model_wts = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping on epoch {epoch + 1} due to lack of improvement in validation accuracy.")
                break

    # Save best model
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
        model_name = f"models/{args.arch}_{'pre' if args.use_preprocessed else 'orig'}.pth"
        torch.save(model.state_dict(), model_name)
        print(f"Best model saved to {model_name}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["v1", "v2", "resnet34", "efficientnet", "densenet", "resnet50"], required=True, help="CNN architecture version")
    parser.add_argument("--use_preprocessed", action="store_true", help="Use preprocessed images")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    args = parser.parse_args()

    # Run the main function
    main(args)