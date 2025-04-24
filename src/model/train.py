import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import os
import csv  # Add this import at the top of the file

from chestxray_cnn import ChestXrayCNN
from chestxray_cnn_v2 import ChestXrayCNNv2  # Assuming you have a second version of the CNN
from dataset.chestxray_dataset import ChestXrayDataset  # Import the dataset class

def get_model(arch, num_classes):
    """
    Returns the model architecture based on the specified version.
    Args:
        arch (str): The architecture version ("v1" or "v2").
        num_classes (int): The number of output classes.
    Returns:
        nn.Module: The selected model.
    """
    if arch == "v1":
        return ChestXrayCNN(num_classes=num_classes)
    elif arch == "v2":
        return ChestXrayCNNv2(num_classes=num_classes)
    else:
        raise ValueError("Unsupported architecture. Use 'v1' or 'v2'.")


def main(args):
    # Set the device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Print the selected device
    if torch.cuda.is_available():
        print("Using GPU for training.")
    else:
        print("GPU not available. Using CPU for training.")

    # Load the training dataset
    train_dataset = ChestXrayDataset(
        csv_file='data/train.csv',
        image_root='data/images',
        cache_dir='data/preprocessed',
        use_preprocessed=args.use_preprocessed,
        target_size=(224, 224),
        train=True
    )

    # Create DataLoader for training
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    # Initialize the model based on the specified architecture
    model = get_model(args.arch, num_classes=7).to(device)

    # Check if the model is successfully loaded into the GPU
    if torch.cuda.is_available() and next(model.parameters()).is_cuda:
        print("Model successfully loaded into GPU.")
    else:
        print("Model is using CPU.")

    # Test if the model, inputs, and labels are loaded into the GPU
    print("Testing if model, inputs, and labels are loaded into GPU...")
    test_inputs, test_labels = next(iter(train_loader))
    test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
    test_outputs = model(test_inputs)
    if torch.cuda.is_available() and test_inputs.is_cuda and test_labels.is_cuda and next(model.parameters()).is_cuda:
        print("Model, inputs, and labels are successfully loaded into GPU.")
    else:
        print("Model, inputs, or labels are not loaded into GPU. Check your setup.")

    # Define the loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    # Define the directory for saving metrics
    metrics_dir = "src/model/training_metrics"
    os.makedirs(metrics_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Define the full path for the CSV file
    csv_file_path = os.path.join(metrics_dir, f"metrics_{args.arch}_{'pre' if args.use_preprocessed else 'orig'}.csv")

    # Open a CSV file to save metrics
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write the header row
        csv_writer.writerow(["Epoch", "Loss", "Accuracy"])

        # Training loop
        for epoch in range(args.epochs):
            model.train()  # Set the model to training mode
            running_loss = 0.0
            correct = 0
            total = 0

            # Iterate through the training DataLoader
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
                # Move inputs and labels to the selected device (GPU or CPU)
                inputs, labels = inputs.to(device), labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)

                # Calculate the loss
                loss = criterion(outputs, labels)

                # Backpropagation
                loss.backward()

                # Update the model weights
                optimizer.step()

                # Accumulate the loss
                running_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)  # Get the predicted class
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            # Update the learning rate scheduler
            scheduler.step()

            # Calculate and print the epoch loss and accuracy
            accuracy = 100 * correct / total
            epoch_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

            # Save metrics to the CSV file
            csv_writer.writerow([epoch + 1, epoch_loss, accuracy])

    print(f"Metrics saved to {csv_file_path}")

    # Save the trained model
    model_name = f"models/{args.arch}_{'pre' if args.use_preprocessed else 'orig'}.pth"
    torch.save(model.state_dict(), model_name)
    print(f"Model saved to {model_name}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["v1", "v2"], required=True, help="CNN architecture version")
    parser.add_argument("--use_preprocessed", action="store_true", help="Use preprocessed images")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    args = parser.parse_args()

    # Run the main function
    main(args)

