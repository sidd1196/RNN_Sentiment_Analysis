"""
Training functions for sentiment classification models.
Supports: Adam, SGD, RMSProp optimizers with optional gradient clipping.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm
from utils import get_device


def train_epoch(model, train_loader, criterion, optimizer, device, grad_clip=None):
    """
    Train model for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training DataLoader
        criterion: Loss function
        optimizer: Optimizer
        device: Device (CPU/CUDA)
        grad_clip: Gradient clipping value (None if not used)
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_texts, batch_labels in tqdm(train_loader, desc="Training", leave=False):
        batch_texts = batch_texts.to(device)
        batch_labels = batch_labels.to(device).unsqueeze(1)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_texts)
        loss = criterion(outputs, batch_labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (if specified)
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def train_model(model, train_loader, test_loader, optimizer_name='adam', 
                learning_rate=0.001, num_epochs=10, grad_clip=None, device=None):
    """
    Train a model and return training history.
    
    Args:
        model: Model to train
        train_loader: Training DataLoader
        test_loader: Test DataLoader (for validation, not used in training)
        optimizer_name: 'adam', 'sgd', or 'rmsprop'
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        grad_clip: Gradient clipping value (None if not used)
        device: Device (CPU/CUDA)
    
    Returns:
        Dictionary with 'train_losses' and 'epoch_times' lists
    """
    if device is None:
        device = get_device()
    
    model = model.to(device)
    
    # Loss function (binary cross-entropy as per requirements)
    criterion = nn.BCELoss()
    
    # Optimizer
    optimizer_name = optimizer_name.lower()
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    # Training history
    history = {
        'train_losses': [],
        'epoch_times': []
    }
    
    print(f"Training with {optimizer_name.upper()} optimizer, LR={learning_rate}")
    if grad_clip:
        print(f"Using gradient clipping with max_norm={grad_clip}")
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, grad_clip)
        
        epoch_time = time.time() - epoch_start
        
        history['train_losses'].append(train_loss)
        history['epoch_times'].append(epoch_time)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f} - Time: {epoch_time:.2f}s")
    
    avg_epoch_time = sum(history['epoch_times']) / len(history['epoch_times'])
    print(f"Average epoch time: {avg_epoch_time:.2f}s")
    
    return history
