import argparse
import torch
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from dataset.chestxray_dataset import ChestXrayDataset
from chestxray_cnn import ChestXrayCNN
from chestxray_cnn_v2 import ChestXrayCNNv2
from tqdm import tqdm
from train import get_model  # Import the get_model function from train.py


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
        tuple: Accuracy, sensitivity, average specificity, AUC, confusion matrix, precision, recall, and F1 score.
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
    acc = accuracy_score(all_labels, all_preds) * 100
    cm = confusion_matrix(all_labels, all_preds)

    # Overall sensitivity (macro-average recall)
    sensitivity = recall_score(all_labels, all_preds, average='macro')

    # Per-class specificity
    specificity = []
    for i in range(len(label_names)):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])  # True Negatives for class i
        fp = cm[:, i].sum() - cm[i, i]  # False Positives for class i
        specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

    # Average specificity (macro-average)
    average_specificity = sum(specificity) / len(specificity)

    # AUC (macro-average)
    auc = roc_auc_score(all_labels, np.eye(len(label_names))[all_preds], average='macro', multi_class='ovr')

    # Per-class metrics
    precision = precision_score(all_labels, all_preds, average=None)
    recall = recall_score(all_labels, all_preds, average=None)
    f1 = f1_score(all_labels, all_preds, average=None)

    # Print overall metrics
    print(f"Accuracy: {acc:.2f}%\nSensitivity (Macro): {sensitivity:.4f}\nAUC: {auc:.4f}\nAverage Specificity: {average_specificity:.4f}")

    return acc, sensitivity, average_specificity, auc, cm, precision, recall, f1


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
    elif "resnet34" in filename:
        arch = "resnet34"
    elif "resnet50" in filename:
        arch = "resnet50"
    elif "efficientnet" in filename:
        arch = "efficientnet"
    elif "densenet" in filename:
        arch = "densenet"
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
    label_names = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 
                   'Mass', 'Nodule', 'No Finding']

    # Load the test dataset
    test_dataset = ChestXrayDataset(
        csv_file='data/test.csv',
        use_preprocessed=use_preprocessed,  # Use preprocessed images if inferred
        train=False  # No augmentations for testing
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Load the model
    model = get_model(arch, num_classes=7)
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))

    # Evaluate the model
    acc, sens, avg_spec, auc, cm, precision, recall, f1 = evaluate_model(model, test_loader, label_names)

    # Ensure the evaluations directory exists inside the models directory
    evaluations_dir = os.path.join(os.path.dirname(args.model_path), "evaluations")
    os.makedirs(evaluations_dir, exist_ok=True)

    # Save overall results to a CSV file in the evaluations directory
    csv_path = os.path.join(evaluations_dir, "evaluation_results.csv")

    # Check if the file already exists
    file_exists = os.path.isfile(csv_path)

    # Open the file in append mode if it exists, otherwise create it
    with open(csv_path, 'a' if file_exists else 'w') as f:
        # Write the header only if the file is being created
        if not file_exists:
            f.write('Model,Accuracy,Sensitivity,Specificity,AUC\n')
        # Append the new row with the evaluation results
        f.write(f"{model_name},{acc:.2f},{sens:.4f},{avg_spec:.4f},{auc:.4f}\n")

    print(f"Evaluation results saved to {csv_path}")

    # Save per-class metrics to CSV
    per_class_metrics_path = os.path.join(evaluations_dir, f"{model_name}_classwise_metrics.csv")
    with open(per_class_metrics_path, 'w') as f:
        f.write("Class,Precision,Recall,F1-Score\n")
        for idx, class_name in enumerate(label_names):
            f.write(f"{class_name},{precision[idx]:.4f},{recall[idx]:.4f},{f1[idx]:.4f}\n")
    print(f"Per-class metrics saved to {per_class_metrics_path}")

    # Save the confusion matrix plot in the evaluations directory
    cm_path = os.path.join(evaluations_dir, f"{model_name}_cm.png")
    plot_confusion_matrix(cm, label_names, cm_path)
    print(f"Confusion matrix saved to {cm_path}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to the trained model .pth file")
    args = parser.parse_args()

    # Run the main function
    main(args)
