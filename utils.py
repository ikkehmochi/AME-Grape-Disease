"""
Contains various utility functions for PyTorch model training and saving.
"""
import sklearn.metrics
import torch
from pathlib import Path
from typing import Tuple
import sklearn
import numpy as np
import matplotlib.pyplot as plt

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)

  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)
  
def evaluate_model(model:torch.nn.Module,
                   dataloader:torch.utils.data.DataLoader,
                   device:torch.device):
    pred_labels=[]
    true_labels=[]
    model.eval()
    with torch.inference_mode():
        for batch, (X,y) in enumerate(dataloader):
            X, y= X.to(device), y.to(device)
            test_pred_logits=model(X)
            test_pred_labels=torch.argmax(test_pred_logits, dim=1)
            pred_labels.extend(test_pred_labels.cpu().numpy())
            true_labels.extend(y.cpu().numpy())
    return np.array(pred_labels), np.array(true_labels)
def plot_CM(classes,
            y_pred,
            y_true):
    cm=sklearn.metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
    disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot()
    plt.title("Prediksi Penyakit Anggur")
    plt.show()

def plot_f1(models,
            models_name,
            dataloader:torch.utils.data.DataLoader,
            device:torch.device):
    f1_scores=[]
    for model in models:
        y_pred, y_true=evaluate_model(model, dataloader, device)
        f1_scores.append(sklearn.metrics.f1_score(y_true=y_true, y_pred=y_pred, average='weighted'))
    plt.figure(figsize=(16,16))
    plt.bar(models_name, f1_scores, color='skyblue')
    plt.xlabel('Models')
    plt.ylabel('F1 Score (Weighted)')
    plt.title('F1 Score Comparison of Models')
    plt.xticks(rotation=45)
    plt.show()