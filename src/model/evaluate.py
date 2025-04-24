import argparse
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from dataset.chestxray_dataset import ChestXrayDataset
from chestxray_cnn import ChestXrayCNN
from chestxray_cnn_v2 import ChestXrayCNNv2
import os
from tqdm import tqdm  # Add this import at the top of the file


def plot_confusion_matrix(cm, labels, save_path):
    """
    Plots the confusion matrix and saves it as an image.
    Args:
        cm (ndarray): Confusion matrix.
        labels (list): List of class labels.
        save_path (str): Path to save the confusion matrix image.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def evaluate_model(model, test_loader, label_names):
    """
    Evaluates the model on the test dataset and calculates metrics.
    Args:
        model (nn.Module): Trained model.
        test_loader (DataLoader): DataLoader for the test dataset.
        label_names (list): List of class labels.
    Returns:
        tuple: Accuracy, sensitivity, specificity, AUC, and confusion matrix.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    # Disable gradient computation for evaluation
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Get the predicted class
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # Calculate metrics
    acc = 100 * correct / total
    cm = confusion_matrix(all_labels, all_preds)
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    auc = roc_auc_score(all_labels, all_preds, average='macro', multi_class='ovr')

    # Print metrics
    print(f"Accuracy: {acc:.2f}%\nSensitivity: {sensitivity:.4f}\nSpecificity: {specificity:.4f}\nAUC: {auc:.4f}")

    return acc, sensitivity, specificity, auc, cm


def infer_args_from_model_path(model_path):
    """
    Infers the architecture and preprocessing flag from the model path.
    Args:
        model_path (str): Path to the trained model file.
    Returns:
        tuple: (arch, use_preprocessed, model_name)
    """
    filename = os.path.basename(model_path)
    if "v1" in filename:
        arch = "v1"
    elif "v2" in filename:
        arch = "v2"
    else:
        raise ValueError("Unable to infer architecture from model path. Ensure 'v1' or 'v2' is in the filename.")

    if "pre" in filename:
        use_preprocessed = True
    elif "orig" in filename:
        use_preprocessed = False
    else:
        raise ValueError("Unable to infer preprocessing flag from model path. Ensure 'pre' or 'orig' is in the filename.")

    # Combine architecture and preprocessing flag for the model name
    model_name = filename.replace(".pth", "")  # e.g., "v1_pre" or "v2_orig"
    return arch, use_preprocessed, model_name


def main(args):
    """
    Main function for evaluating the model.
    Args:
        args: Command-line arguments.
    """
    # Infer architecture, preprocessing flag, and model name from the model path
    arch, use_preprocessed, model_name = infer_args_from_model_path(args.model_path)

    # Define the class labels
    label_names = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'No Finding']

    # Load the test dataset
    test_dataset = ChestXrayDataset(
        csv_file='data/test.csv',
        image_root='data/images',
        cache_dir='data/preprocessed',
        use_preprocessed=use_preprocessed,  # Use preprocessed images if inferred
        target_size=(224, 224),
        train=False  # No augmentations for testing
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Load the model
    model_cls = ChestXrayCNN if arch == 'v1' else ChestXrayCNNv2
    model = model_cls(num_classes=7)
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))

    # Evaluate the model
    acc, sens, spec, auc, cm = evaluate_model(model, test_loader, label_names)

    # Ensure the evaluations directory exists inside the models directory
    evaluations_dir = os.path.join(os.path.dirname(args.model_path), "evaluations")
    os.makedirs(evaluations_dir, exist_ok=True)

    # Save results to a CSV file in the evaluations directory
    csv_path = os.path.join(evaluations_dir, "evaluation_results.csv")
    with open(csv_path, 'w') as f:
        f.write('Model,Accuracy,Sensitivity,Specificity,AUC\n')
        f.write(f"{model_name},{acc:.2f},{sens:.4f},{spec:.4f},{auc:.4f}\n")
    print(f"Evaluation results saved to {csv_path}")

    # Save the confusion matrix plot in the evaluations directory
    cm_path = os.path.join(evaluations_dir, f"{model_name}.png")
    plot_confusion_matrix(cm, label_names, cm_path)
    print(f"Confusion matrix saved to {cm_path}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to the trained model .pth file")
    args = parser.parse_args()

    # Run the main function
    main(args)
