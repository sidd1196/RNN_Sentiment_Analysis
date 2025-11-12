"""
Evaluation functions for sentiment classification models.
Computes: Accuracy and F1-score (macro).
"""
import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from utils import get_device


def evaluate_model(model, test_loader, device=None):
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        test_loader: Test DataLoader
        device: Device (CPU/CUDA)
    
    Returns:
        Dictionary with 'accuracy' and 'f1_score' (macro)
    """
    if device is None:
        device = get_device()
    
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_texts, batch_labels in tqdm(test_loader, desc="Evaluating", leave=False):
            batch_texts = batch_texts.to(device)
            batch_labels = batch_labels.cpu().numpy()
            
            outputs = model(batch_texts)
            predictions = (outputs.cpu().numpy() > 0.5).astype(int).flatten()
            
            all_predictions.extend(predictions)
            all_labels.extend(batch_labels.astype(int))
    
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='macro')
    
    return {
        'accuracy': accuracy,
        'f1_score': f1
    }
