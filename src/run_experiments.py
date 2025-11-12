"""
Main experiment runner - Systematic 50 experiments covering all models, sequence lengths, and stability strategies.
Uses recommended activation functions and optimizers for each architecture.
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import json

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess import prepare_data
from models import SentimentRNN
from train import train_model
from evaluate import evaluate_model
from utils import set_seeds, get_device, get_hardware_info


def run_experiments():
    """Run systematic experimental configuration (50 experiments covering all combinations)."""
    # Set seeds for reproducibility
    set_seeds(42)
    
    # Get hardware info
    hardware_info = get_hardware_info()
    print("Hardware Information:")
    for key, value in hardware_info.items():
        print(f"  {key}: {value}")
    print()
    
    # Define systematic experiments covering all models, sequence lengths, and stability strategies
    # Using mixed activations: Each model type (RNN, LSTM, BiLSTM) has both Tanh and ReLU experiments
    # Testing all optimizers (Adam, SGD, RMSProp) for comprehensive coverage
    experiments = [
        # -------------------- RNN Experiments (Mixed: Tanh and ReLU) --------------------
        # Vary sequence length (SGD, clipping) - Tanh
        {"model": "rnn", "activation": "tanh", "seq_len": 25, "optimizer": "SGD", "strategy": "clipping"},
        {"model": "rnn", "activation": "tanh", "seq_len": 50, "optimizer": "SGD", "strategy": "clipping"},
        {"model": "rnn", "activation": "tanh", "seq_len": 100, "optimizer": "SGD", "strategy": "clipping"},
        # Vary sequence length (SGD, no clipping) - Tanh
        {"model": "rnn", "activation": "tanh", "seq_len": 25, "optimizer": "SGD", "strategy": "no clipping"},
        {"model": "rnn", "activation": "tanh", "seq_len": 50, "optimizer": "SGD", "strategy": "no clipping"},
        {"model": "rnn", "activation": "tanh", "seq_len": 100, "optimizer": "SGD", "strategy": "no clipping"},
        # Vary sequence length (Adam, clipping) - Sigmoid (seq=25), Tanh (seq=50, 100)
        {"model": "rnn", "activation": "sigmoid", "seq_len": 25, "optimizer": "Adam", "strategy": "clipping"},
        {"model": "rnn", "activation": "tanh", "seq_len": 50, "optimizer": "Adam", "strategy": "clipping"},
        {"model": "rnn", "activation": "tanh", "seq_len": 100, "optimizer": "Adam", "strategy": "clipping"},
        # Vary sequence length (Adam, no clipping) - ReLU
        {"model": "rnn", "activation": "relu", "seq_len": 25, "optimizer": "Adam", "strategy": "no clipping"},
        {"model": "rnn", "activation": "relu", "seq_len": 50, "optimizer": "Adam", "strategy": "no clipping"},
        {"model": "rnn", "activation": "relu", "seq_len": 100, "optimizer": "Adam", "strategy": "no clipping"},
        # Vary sequence length (RMSProp, clipping) - ReLU
        {"model": "rnn", "activation": "relu", "seq_len": 25, "optimizer": "RMSProp", "strategy": "clipping"},
        {"model": "rnn", "activation": "relu", "seq_len": 50, "optimizer": "RMSProp", "strategy": "clipping"},
        {"model": "rnn", "activation": "relu", "seq_len": 100, "optimizer": "RMSProp", "strategy": "clipping"},
        # Vary sequence length (RMSProp, no clipping) - ReLU
        {"model": "rnn", "activation": "relu", "seq_len": 25, "optimizer": "RMSProp", "strategy": "no clipping"},
        {"model": "rnn", "activation": "relu", "seq_len": 50, "optimizer": "RMSProp", "strategy": "no clipping"},
        {"model": "rnn", "activation": "relu", "seq_len": 100, "optimizer": "RMSProp", "strategy": "no clipping"},
        
        # -------------------- LSTM Experiments (Mixed: ReLU and Tanh) --------------------
        # Vary sequence length (SGD, clipping) - ReLU
        {"model": "lstm", "activation": "relu", "seq_len": 25, "optimizer": "SGD", "strategy": "clipping"},
        {"model": "lstm", "activation": "relu", "seq_len": 50, "optimizer": "SGD", "strategy": "clipping"},
        {"model": "lstm", "activation": "relu", "seq_len": 100, "optimizer": "SGD", "strategy": "clipping"},
        # Vary sequence length (SGD, no clipping) - ReLU
        {"model": "lstm", "activation": "relu", "seq_len": 25, "optimizer": "SGD", "strategy": "no clipping"},
        {"model": "lstm", "activation": "relu", "seq_len": 50, "optimizer": "SGD", "strategy": "no clipping"},
        {"model": "lstm", "activation": "relu", "seq_len": 100, "optimizer": "SGD", "strategy": "no clipping"},
        # Vary sequence length (Adam, clipping) - ReLU
        {"model": "lstm", "activation": "relu", "seq_len": 25, "optimizer": "Adam", "strategy": "clipping"},
        {"model": "lstm", "activation": "relu", "seq_len": 50, "optimizer": "Adam", "strategy": "clipping"},
        {"model": "lstm", "activation": "relu", "seq_len": 100, "optimizer": "Adam", "strategy": "clipping"},
        # Vary sequence length (Adam, no clipping) - Tanh
        {"model": "lstm", "activation": "tanh", "seq_len": 25, "optimizer": "Adam", "strategy": "no clipping"},
        {"model": "lstm", "activation": "tanh", "seq_len": 50, "optimizer": "Adam", "strategy": "no clipping"},
        {"model": "lstm", "activation": "tanh", "seq_len": 100, "optimizer": "Adam", "strategy": "no clipping"},
        # Vary sequence length (RMSProp, clipping) - Sigmoid (seq=25), Tanh (seq=50, 100)
        {"model": "lstm", "activation": "sigmoid", "seq_len": 25, "optimizer": "RMSProp", "strategy": "clipping"},
        {"model": "lstm", "activation": "tanh", "seq_len": 50, "optimizer": "RMSProp", "strategy": "clipping"},
        {"model": "lstm", "activation": "tanh", "seq_len": 100, "optimizer": "RMSProp", "strategy": "clipping"},
        # Vary sequence length (RMSProp, no clipping) - Tanh
        {"model": "lstm", "activation": "tanh", "seq_len": 25, "optimizer": "RMSProp", "strategy": "no clipping"},
        {"model": "lstm", "activation": "tanh", "seq_len": 50, "optimizer": "RMSProp", "strategy": "no clipping"},
        
        # -------------------- BiLSTM Experiments (Mixed: Tanh and ReLU) --------------------
        # Vary sequence length (SGD, clipping) - Tanh
        {"model": "bilstm", "activation": "tanh", "seq_len": 25, "optimizer": "SGD", "strategy": "clipping"},
        {"model": "bilstm", "activation": "tanh", "seq_len": 50, "optimizer": "SGD", "strategy": "clipping"},
        {"model": "bilstm", "activation": "tanh", "seq_len": 100, "optimizer": "SGD", "strategy": "clipping"},
        # Vary sequence length (SGD, no clipping) - Tanh
        {"model": "bilstm", "activation": "tanh", "seq_len": 25, "optimizer": "SGD", "strategy": "no clipping"},
        {"model": "bilstm", "activation": "tanh", "seq_len": 50, "optimizer": "SGD", "strategy": "no clipping"},
        {"model": "bilstm", "activation": "tanh", "seq_len": 100, "optimizer": "SGD", "strategy": "no clipping"},
        # Vary sequence length (Adam, clipping) - Sigmoid (seq=25, 50), Tanh (seq=100)
        {"model": "bilstm", "activation": "sigmoid", "seq_len": 25, "optimizer": "Adam", "strategy": "clipping"},
        {"model": "bilstm", "activation": "sigmoid", "seq_len": 50, "optimizer": "Adam", "strategy": "clipping"},
        {"model": "bilstm", "activation": "tanh", "seq_len": 100, "optimizer": "Adam", "strategy": "clipping"},
        # Vary sequence length (Adam, no clipping) - ReLU
        {"model": "bilstm", "activation": "relu", "seq_len": 25, "optimizer": "Adam", "strategy": "no clipping"},
        {"model": "bilstm", "activation": "relu", "seq_len": 50, "optimizer": "Adam", "strategy": "no clipping"},
        {"model": "bilstm", "activation": "relu", "seq_len": 100, "optimizer": "Adam", "strategy": "no clipping"},
        # Vary sequence length (RMSProp, clipping) - ReLU
        {"model": "bilstm", "activation": "relu", "seq_len": 25, "optimizer": "RMSProp", "strategy": "clipping"},
        {"model": "bilstm", "activation": "relu", "seq_len": 50, "optimizer": "RMSProp", "strategy": "clipping"},
        {"model": "bilstm", "activation": "relu", "seq_len": 100, "optimizer": "RMSProp", "strategy": "clipping"},
        # Vary sequence length (RMSProp, no clipping) - ReLU
        {"model": "bilstm", "activation": "relu", "seq_len": 25, "optimizer": "RMSProp", "strategy": "no clipping"},
        {"model": "bilstm", "activation": "relu", "seq_len": 50, "optimizer": "RMSProp", "strategy": "no clipping"},
    ]
    
    total_experiments = len(experiments)
    
    print(f"Total experiments to run: {total_experiments}")
    print(f"Coverage:")
    print(f"  - Models: RNN (Tanh & ReLU), LSTM (ReLU & Tanh), BiLSTM (Tanh & ReLU)")
    print(f"  - Sequence Lengths: 25, 50, 100")
    print(f"  - Optimizers: Adam, SGD, RMSProp")
    print(f"  - Stability Strategies: Clipping, No Clipping")
    print()
    
    # Fixed hyperparameters
    embedding_dim = 100
    hidden_size = 64
    num_layers = 2
    dropout = 0.3
    batch_size = 32
    num_epochs = 5
    max_vocab_size = 10000
    
    # CSV path for data loading
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_root, 'data', 'IMDB Dataset.csv')
    
    # Learning rates for different optimizers (default values)
    learning_rates = {
        'Adam': 0.001,      # Default for Adam
        'SGD': 0.01,        # Default for SGD (higher)
        'RMSProp': 0.001    # Default for RMSProp
    }
    
    print(f"Experiments breakdown:")
    rnn_count = sum(1 for exp in experiments if exp["model"] == "rnn")
    lstm_count = sum(1 for exp in experiments if exp["model"] == "lstm")
    bilstm_count = sum(1 for exp in experiments if exp["model"] == "bilstm")
    print(f"  RNN: {rnn_count} experiments")
    print(f"  LSTM: {lstm_count} experiments")
    print(f"  BiLSTM: {bilstm_count} experiments")
    print()
    
    # Results storage
    results = []
    data_cache = {}
    vocab_cache = None
    
    print("=" * 80)
    print("Starting Experiments")
    print("=" * 80)
    
    experiment_num = 0
    
    for exp_config in experiments:
        experiment_num += 1
        
        # Extract experiment parameters
        model_type = exp_config["model"]
        activation = exp_config["activation"]
        seq_len = exp_config["seq_len"]
        optimizer = exp_config["optimizer"]
        strategy = exp_config["strategy"]
        
        # Map parameters to model configuration
        if model_type == "rnn":
            rnn_type = "RNN"
            bidirectional = False
            arch_name = "RNN"
        elif model_type == "lstm":
            rnn_type = "LSTM"
            bidirectional = False
            arch_name = "LSTM"
        elif model_type == "bilstm":
            rnn_type = "LSTM"
            bidirectional = True
            arch_name = "Bidirectional LSTM"
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Map activation (handle special cases like ReLU)
        activation_map = {
            "relu": "ReLU",
            "tanh": "Tanh",
            "sigmoid": "Sigmoid"
        }
        activation_capitalized = activation_map.get(activation.lower(), activation.capitalize())
        
        # Map optimizer (ensure consistent capitalization)
        optimizer_lower = optimizer.lower()
        if optimizer_lower == "rmsprop":
            optimizer_name = "RMSProp"
        elif optimizer_lower == "adam":
            optimizer_name = "Adam"
        elif optimizer_lower == "sgd":
            optimizer_name = "SGD"
        else:
            optimizer_name = optimizer.capitalize()
        
        # Get learning rate for this optimizer
        learning_rate = learning_rates[optimizer_name]
        
        # Map strategy to gradient clipping
        grad_clip = 1.0 if strategy == "clipping" else None
        
        # Print experiment info (simplified)
        print(f"\n[{experiment_num}/{total_experiments}] Model={arch_name}, Activation={activation_capitalized}, "
              f"Optimizer={optimizer_name} (LR={learning_rate}), SeqLen={seq_len}, "
              f"GradClip={'Yes' if grad_clip else 'No'}")
        
        # Prepare data for this sequence length (cache to avoid reloading)
        if seq_len not in data_cache:
            train_loader, test_loader, token_to_id, _, _ = prepare_data(
                csv_path=csv_path,
                max_vocab_size=max_vocab_size,
                max_length=seq_len,
                batch_size=batch_size,
                use_existing_vocab=vocab_cache
            )
            data_cache[seq_len] = (train_loader, test_loader)
            if vocab_cache is None:
                vocab_cache = token_to_id
        
        train_loader, test_loader = data_cache[seq_len]
        
        # Create model
        model = SentimentRNN(
            vocab_size=len(vocab_cache),
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            rnn_type=rnn_type,
            activation=activation_capitalized,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        # Train model
        history = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            grad_clip=grad_clip
        )
        
        # Evaluate model
        metrics = evaluate_model(model, test_loader)
        
        # Store results
        avg_epoch_time = sum(history['epoch_times']) / len(history['epoch_times'])
        
        result = {
            'Model': arch_name,
            'Activation': activation_capitalized,
            'Optimizer': optimizer_name,
            'Seq Length': seq_len,
            'Grad Clipping': 'Yes' if grad_clip else 'No',
            'Accuracy': metrics['accuracy'],
            'F1': metrics['f1_score'],
            'Epoch Time (s)': avg_epoch_time,
            'Final Train Loss': history['train_losses'][-1],
            'Train Losses': history['train_losses']
        }
        results.append(result)
        
        print(f"  Results: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}, "
              f"Avg Epoch Time={avg_epoch_time:.2f}s")
    
    # Save results
    print(f"\n{'='*80}")
    print("Saving results...")
    print(f"{'='*80}")
    
    # Create summary table matching the required format
    results_df = pd.DataFrame([
        {
            'Model': r['Model'],
            'Activation': r['Activation'],
            'Optimizer': r['Optimizer'],
            'Seq Length': r['Seq Length'],
            'Grad Clipping': r['Grad Clipping'],
            'Accuracy': r['Accuracy'],
            'F1': r['F1'],
            'Epoch Time (s)': r['Epoch Time (s)']
        }
        for r in results
    ])
    
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save CSV with exact column order as specified
    csv_path = os.path.join(results_dir, 'metrics.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Save detailed results
    with open(os.path.join(results_dir, 'detailed_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to {os.path.join(results_dir, 'detailed_results.json')}")
    
    # Generate plots
    print("\nGenerating plots...")
    generate_plots(results, hardware_info)
    
    print(f"\n{'='*80}")
    print("All experiments completed!")
    print(f"{'='*80}")
    
    return results_df


def generate_plots(results, hardware_info):
    """Generate required plots: Accuracy/F1 vs Sequence Length, Training Loss vs Epochs (best/worst)."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plots_dir = os.path.join(project_root, 'results', 'plots')
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(plots_dir, exist_ok=True)
    
    df = pd.DataFrame([
        {
            'Model': r['Model'],
            'Activation': r['Activation'],
            'Optimizer': r['Optimizer'],
            'Seq Length': r['Seq Length'],
            'Grad Clipping': r['Grad Clipping'],
            'Accuracy': r['Accuracy'],
            'F1': r['F1'],
            'Epoch Time (s)': r['Epoch Time (s)'],
            'Train Losses': r['Train Losses']
        }
        for r in results
    ])
    
    # Plot 1: Accuracy/F1 vs. Sequence Length (by Model)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    seq_lengths = sorted(df['Seq Length'].unique())
    colors = {'RNN': 'blue', 'LSTM': 'green', 'Bidirectional LSTM': 'red'}
    markers = {'RNN': 'o', 'LSTM': 's', 'Bidirectional LSTM': '^'}
    
    for model in sorted(df['Model'].unique()):
        model_data = df[df['Model'] == model]
        accuracies = [model_data[model_data['Seq Length'] == sl]['Accuracy'].mean() 
                     for sl in seq_lengths]
        f1_scores = [model_data[model_data['Seq Length'] == sl]['F1'].mean() 
                    for sl in seq_lengths]
        
        ax1.plot(seq_lengths, accuracies, marker=markers[model], label=model, 
                color=colors[model], linewidth=2, markersize=8)
        ax2.plot(seq_lengths, f1_scores, marker=markers[model], label=model,
                color=colors[model], linewidth=2, markersize=8)
    
    ax1.set_xlabel('Sequence Length', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Accuracy vs. Sequence Length', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(seq_lengths)
    
    ax2.set_xlabel('Sequence Length', fontsize=12)
    ax2.set_ylabel('F1-Score', fontsize=12)
    ax2.set_title('F1-Score vs. Sequence Length', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(seq_lengths)
    
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, 'accuracy_f1_vs_seq_length.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()
    
    # Plot 2: Training Loss vs. Epochs (best and worst models)
    best_idx = df['Accuracy'].idxmax()
    worst_idx = df['Accuracy'].idxmin()
    
    best_result = results[best_idx]
    worst_result = results[worst_idx]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs_best = range(1, len(best_result['Train Losses']) + 1)
    epochs_worst = range(1, len(worst_result['Train Losses']) + 1)
    
    ax.plot(epochs_best, best_result['Train Losses'], marker='o', label='Best Model', 
            linewidth=2.5, markersize=8, color='green')
    ax.plot(epochs_worst, worst_result['Train Losses'], marker='s', label='Worst Model',
            linewidth=2.5, markersize=8, color='red')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title('Training Loss vs. Epochs (Best vs. Worst Models)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    best_label = (f"Best: {best_result['Model']}, {best_result['Activation']}, "
                  f"{best_result['Optimizer']}, Seq={best_result['Seq Length']}, "
                  f"Clip={best_result['Grad Clipping']}, Acc={best_result['Accuracy']:.3f}")
    worst_label = (f"Worst: {worst_result['Model']}, {worst_result['Activation']}, "
                   f"{worst_result['Optimizer']}, Seq={worst_result['Seq Length']}, "
                   f"Clip={worst_result['Grad Clipping']}, Acc={worst_result['Accuracy']:.3f}")
    
    ax.text(0.02, 0.98, best_label, transform=ax.transAxes, 
            verticalalignment='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax.text(0.02, 0.88, worst_label, transform=ax.transAxes, 
            verticalalignment='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, 'training_loss_best_worst.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()
    
    # Save hardware info
    hardware_info_path = os.path.join(results_dir, 'hardware_info.json')
    with open(hardware_info_path, 'w') as f:
        json.dump(hardware_info, f, indent=2)
    print(f"Saved: {hardware_info_path}")


if __name__ == '__main__':
    results_df = run_experiments()
    print("\nSummary of Results:")
    print(results_df.to_string())
