# Name - Siddharth Pathania
# UID - 121291592



# Comparative Analysis of RNN Architectures for Sentiment Classification

This project implements and evaluates multiple Recurrent Neural Network (RNN) architectures for sentiment classification on the IMDb Movie Review Dataset.

## Project Structure

```
├── data/                    # Dataset directory (IMDb reviews)
├── src/
│   ├── preprocess.py        # Data preprocessing and loading
│   ├── models.py            # RNN model architectures
│   ├── train.py             # Training functions
│   ├── evaluate.py          # Evaluation functions
│   ├── utils.py             # Utility functions (seeds, device)
│   └── run_experiments.py   # Main experiment runner
├── results/
│   ├── metrics.csv          # Results table
│   ├── detailed_results.json # Detailed results with training losses
│   ├── hardware_info.json   # Hardware information
│   └── plots/               # Generated plots
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Setup Instructions

### Python Version
- Python 3.8 or higher

### Installation

1. Clone or download this repository.

2. Create and activate the virtual environment:
```bash
# Create virtual environment
python3 -m venv msml_641_hw3_env

# Activate the virtual environment
# On macOS/Linux:
source msml_641_hw3_env/bin/activate

# On Windows:
# msml_641_hw3_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: 
- Always activate the virtual environment before running scripts
- On macOS, use `python3` instead of `python` if `python` is not found
- To deactivate the virtual environment, simply type `deactivate`

### Dataset Setup

The code supports multiple methods to load the IMDb dataset (in order of priority):

**Option 1: CSV file (automatic if available)**
- If `IMDB Dataset.csv` exists in the project root, it will be used automatically
- The CSV should have columns: `review` and `sentiment` (positive/negative)
- The dataset will be automatically split 50/50 for training and testing

**Option 2: Automatic download**
- The code will automatically try to download using `torchtext` or `datasets` library
- No manual setup required

**Option 3: Manual download (directory structure)**
1. Download the IMDb dataset from: https://ai.stanford.edu/~amaas/data/sentiment/
2. Extract the archive
3. Place the extracted files in the `data/` directory with the following structure:
```
data/
  train/
    pos/    (12,500 positive review files)
    neg/    (12,500 negative review files)
  test/
    pos/    (12,500 positive review files)
    neg/    (12,500 negative review files)
```

## Quick Start

1. **Activate the virtual environment:**
   ```bash
   source msml_641_hw3_env/bin/activate
   ```

2. **Test with a single experiment (recommended first):**
   ```bash
   cd src
   python test_single_experiment.py
   ```

3. **Run all experiments:**
   ```bash
   cd src
   python run_experiments.py
   ```

## Usage

### Running All Experiments

**Important**: Make sure the virtual environment is activated first!

To run all experimental configurations:

```bash
# Activate virtual environment (if not already activated)
source msml_641_hw3_env/bin/activate

# Navigate to src directory
cd src

# Run experiments
python run_experiments.py
```

**Note**: When the virtual environment is activated, you can use `python` instead of `python3`.

This will:
- Load and preprocess the IMDb dataset
- Train models with all combinations of:
  - Architectures: RNN, LSTM, Bidirectional LSTM
  - Activations: Sigmoid, ReLU, Tanh
  - Optimizers: Adam, SGD, RMSProp
  - Sequence lengths: 25, 50, 100
  - Gradient clipping: Yes/No
- Generate results table (`results/metrics.csv`)
- Generate plots:
  - Accuracy/F1 vs. Sequence Length
  - Training Loss vs. Epochs (best vs. worst models)

### Expected Runtime

- **Total experiments**: 3 architectures × 3 activations × 3 optimizers × 3 sequence lengths × 2 gradient clipping options = **162 experiments**
- **Each experiment**: ~10 epochs
- **Estimated total time**: Several hours to days depending on hardware (CPU-only will be slower)

**Note**: For faster testing, you can modify `run_experiments.py` to run a subset of experiments.

### Output Files

After running experiments, you will find:

1. **`results/metrics.csv`**: Summary table with all results
   - Columns: Model, Activation, Optimizer, Seq Length, Grad Clipping, Accuracy, F1, Epoch Time (s)

2. **`results/detailed_results.json`**: Detailed results including training loss history

3. **`results/hardware_info.json`**: Hardware information for reproducibility

4. **`results/plots/accuracy_f1_vs_seq_length.png`**: Plot comparing accuracy and F1-score across sequence lengths

5. **`results/plots/training_loss_comparison.png`**: Training loss curves for best and worst models

## Model Configuration

### Fixed Hyperparameters
- Embedding dimension: 100
- Hidden size: 64
- Number of layers: 2
- Dropout: 0.3
- Batch size: 32
- Learning rate: 0.001
- Number of epochs: 10
- Vocabulary size: 10,000 (top frequent words)
- Loss function: Binary Cross-Entropy
- Output activation: Sigmoid

### Variations Tested
- **Architectures**: RNN, LSTM, Bidirectional LSTM
- **Activations**: Sigmoid, ReLU, Tanh
- **Optimizers**: Adam, SGD, RMSProp
- **Sequence lengths**: 25, 50, 100 words
- **Gradient clipping**: None, 1.0 (max_norm)

## Reproducibility

The code sets random seeds for reproducibility:
- PyTorch: `torch.manual_seed(42)`
- NumPy: `np.random.seed(42)`
- Python random: `random.seed(42)`

Hardware information is automatically recorded in `results/hardware_info.json`.

## Code Components

### `preprocess.py`
- `preprocess_text()`: Lowercases, removes punctuation, tokenizes text
- `build_vocabulary()`: Builds vocabulary from top N frequent words
- `load_imdb_data()`: Loads IMDb dataset (multiple methods)
- `prepare_data()`: Complete data preparation pipeline

### `models.py`
- `SentimentRNN`: Main model class supporting RNN, LSTM, Bidirectional LSTM with configurable activations

### `train.py`
- `train_epoch()`: Training for one epoch
- `train_model()`: Complete training loop with optimizer and gradient clipping support

### `evaluate.py`
- `evaluate_model()`: Evaluates model and returns accuracy and F1-score

### `utils.py`
- `set_seeds()`: Sets random seeds for reproducibility
- `get_device()`: Gets available device (CPU/CUDA)
- `get_hardware_info()`: Collects hardware information

### `run_experiments.py`
- Main script that runs all experimental configurations
- Generates results tables and plots

## Troubleshooting

### Dataset Loading Issues
If you encounter dataset loading errors:
1. Ensure `torchtext` or `datasets` is installed: `pip install torchtext datasets`
2. Or manually download and place dataset in `data/` directory (see Dataset Setup)

### Memory Issues
If you run out of memory:
- Reduce batch size in `run_experiments.py`
- Process experiments in smaller batches
- Use CPU if GPU memory is limited

### Long Runtime
- The full experiment suite takes many hours
- Consider running a subset of experiments for testing
- Modify the experiment loops in `run_experiments.py` to test specific configurations



