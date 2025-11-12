"""
Data preprocessing for IMDb movie review dataset.
Implements: lowercase, remove punctuation, tokenize, vocabulary building, padding/truncation.
"""
import re
import os
import csv
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch


class IMDBDataset(Dataset):
    """PyTorch Dataset for IMDb reviews."""
    
    def __init__(self, texts, labels, token_to_id, max_length):
        self.texts = texts
        self.labels = labels
        self.token_to_id = token_to_id
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Convert tokens to sequence of token IDs
        sequence = [self.token_to_id.get(token, 0) for token in text]
        
        # Pad or truncate to max_length
        if len(sequence) < self.max_length:
            sequence = sequence + [0] * (self.max_length - len(sequence))
        else:
            sequence = sequence[:self.max_length]
        
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(label, dtype=torch.float32)


def preprocess_text(text):
    """
    Preprocess text: lowercase, remove punctuation, tokenize.
    
    Args:
        text: Raw text string
    
    Returns:
        List of tokens
    """
    # Lowercase
    text = text.lower()
    
    # Remove punctuation and special characters (keep only alphanumeric and spaces)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # Tokenize (split on whitespace)
    tokens = text.split()
    
    return tokens


def build_vocabulary(texts, max_vocab_size=10000):
    """
    Build vocabulary from texts, keeping top max_vocab_size most frequent words.
    
    Args:
        texts: List of tokenized texts (each is a list of tokens)
        max_vocab_size: Maximum vocabulary size
    
    Returns:
        token_to_id: Dictionary mapping tokens to IDs
        id_to_token: Dictionary mapping IDs to tokens
    """
    # Count word frequencies
    word_counts = Counter()
    for text in texts:
        word_counts.update(text)
    
    # Get top max_vocab_size words (reserve 0 for <UNK> and padding)
    most_common = word_counts.most_common(max_vocab_size - 1)
    
    # Create mappings
    token_to_id = {'<UNK>': 0}  # 0 reserved for unknown words and padding
    id_to_token = {0: '<UNK>'}
    
    for idx, (token, _) in enumerate(most_common, start=1):
        token_to_id[token] = idx
        id_to_token[idx] = token
    
    return token_to_id, id_to_token


def load_imdb_data_from_csv(csv_path):
    """
    Load IMDb dataset from CSV file.
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        train_texts, train_labels, test_texts, test_labels
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}. Please ensure the CSV file exists.")
    
    print("Loading IMDb dataset from CSV file...")
    all_texts = []
    all_labels = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            review = row['review']
            sentiment = row['sentiment'].lower()
            
            # Clean HTML tags
            review = re.sub(r'<[^>]+>', '', review)
            review = review.replace('<br />', ' ').replace('<br/>', ' ')
            
            all_texts.append(review)
            all_labels.append(1 if sentiment == 'positive' else 0)
    
    # Split 50/50 for train/test (as per requirements)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        all_texts, all_labels, test_size=0.5, random_state=42, stratify=all_labels
    )
    
    print(f"Loaded {len(train_texts)} training and {len(test_texts)} test samples from CSV")
    return train_texts, train_labels, test_texts, test_labels


def load_imdb_data(data_dir='data'):
    """
    Load IMDb dataset from CSV file (legacy function for backward compatibility).
    
    Args:
        data_dir: Directory containing data
    
    Returns:
        train_texts, train_labels, test_texts, test_labels
    """
    csv_path = os.path.join(data_dir, 'IMDB Dataset.csv')
    return load_imdb_data_from_csv(csv_path)


def prepare_data(csv_path=None, data_dir=None, max_vocab_size=10000, max_length=100, batch_size=32, 
                 use_existing_vocab=None):
    """
    Complete data preparation pipeline:
    1. Load IMDb dataset
    2. Preprocess texts (lowercase, remove punctuation, tokenize)
    3. Build vocabulary (top max_vocab_size words)
    4. Create DataLoaders with padding/truncation to max_length
    
    Args:
        csv_path: Direct path to CSV file (takes precedence over data_dir)
        data_dir: Directory containing data (default: 'data' relative to project root)
        max_vocab_size: Maximum vocabulary size (default: 10000)
        max_length: Sequence length for padding/truncation (default: 100)
        batch_size: Batch size for DataLoader (default: 32)
        use_existing_vocab: Use existing vocabulary instead of building new one
    
    Returns:
        train_loader, test_loader, token_to_id, id_to_token, vocab_size
    """
    # Determine CSV path
    if csv_path is None:
        if data_dir is None:
            current_file = os.path.abspath(__file__)
            src_dir = os.path.dirname(current_file)
            project_root = os.path.dirname(src_dir)
            data_dir = os.path.join(project_root, 'data')
        csv_path = os.path.join(data_dir, 'IMDB Dataset.csv')
    
    # Load data
    print("Loading IMDb dataset...")
    train_texts, train_labels, test_texts, test_labels = load_imdb_data_from_csv(csv_path)
    
    # Preprocess texts
    print("Preprocessing texts...")
    train_texts_processed = [preprocess_text(text) for text in train_texts]
    test_texts_processed = [preprocess_text(text) for text in test_texts]
    
    # Build vocabulary from training data
    if use_existing_vocab is None:
        print(f"Building vocabulary (max size: {max_vocab_size})...")
        token_to_id, id_to_token = build_vocabulary(train_texts_processed, max_vocab_size)
    else:
        token_to_id = use_existing_vocab
        id_to_token = {v: k for k, v in use_existing_vocab.items()}
    
    vocab_size = len(token_to_id)
    print(f"Vocabulary size: {vocab_size}")
    
    # Create datasets
    train_dataset = IMDBDataset(train_texts_processed, train_labels, token_to_id, max_length)
    test_dataset = IMDBDataset(test_texts_processed, test_labels, token_to_id, max_length)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Print statistics
    avg_train_length = np.mean([len(text) for text in train_texts_processed])
    avg_test_length = np.mean([len(text) for text in test_texts_processed])
    print(f"Average training review length: {avg_train_length:.2f} tokens")
    print(f"Average test review length: {avg_test_length:.2f} tokens")
    
    return train_loader, test_loader, token_to_id, id_to_token, vocab_size
