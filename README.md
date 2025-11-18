# EEG-Based Alzheimer's Disease Classification

## Overview
This project implements a deep learning pipeline for classifying EEG recordings into three categories:
- **HC** (Healthy Control)
- **MCI** (Mild Cognitive Impairment)
- **AD** (Alzheimer's Disease)

The model uses a 1D ResNet architecture with age integration to analyze 22-channel EEG signals.

## Dataset
- **Source**: `data/balanced_subset` (CSV files with EEG recordings)
- **Annotations**: `data/annotation.json` (contains patient metadata)
- **Total Samples**: 690+ EEG recordings
- **Class Distribution**: Balanced across HC, MCI, and AD
- **EEG Channels**: 22 channels
- **Sequence Length**: 500 timepoints per sample

## Model Architecture

### ResNet1D with Age Integration
- **Input**: 22-channel EEG signal + age feature
- **Architecture**: 
  - Bandpass filtering (1-40 Hz)
  - 1D ResNet backbone with residual blocks
  - Age feature concatenation in fully connected layers
  - Output: 3-class classification (HC/MCI/AD)

### Key Features
- **Preprocessing**: 
  - Bandpass filtering (1-40 Hz) using Butterworth filter
  - Signal normalization using dataset statistics
  - Age normalization
  - Padding/truncation to fixed length (500 samples)
  
- **Transfer Learning**: 
  - Loads pre-trained checkpoint from `checkpoints/checkpoint.pt`
  - Fine-tunes on balanced subset
  - Two-stage training: classifier-only → full model

- **Class Imbalance Handling**: 
  - Weighted cross-entropy loss
  - Stratified train/test split (80/20)

## Project Structure
```
Team32_AFML/
├── train.ipynb              # Main training notebook
├── data/
│   ├── balanced_subset/     # EEG CSV files
│   └── annotation.json      # Patient metadata (age, symptoms)
├── checkpoints/
│   ├── checkpoint.pt        # Pre-trained model
│   ├── best_stage1_model.pt # Fine-tuned classifier
│   └── final_loaded_model.pt# Best trained model
├── models/
│   └── resnet_1d.py        # ResNet1D model definition
└── README.md               # This file
```

## Setup

### Requirements
```bash
pip install torch numpy scipy scikit-learn matplotlib seaborn
```

### Data Preparation
1. Place EEG CSV files in `data/balanced_subset/`
2. Ensure `data/annotation.json` contains patient metadata with format:
   ```json
   {
     "data": [
       {
         "serial": "00001",
         "age": 65,
         "symptom": ["mci"]
       }
     ]
   }
   ```

## Usage

### Training
Open and run `train.ipynb` in Jupyter or VS Code. The notebook includes:

1. **Data Loading & Preprocessing**
   - Calculates dataset statistics from training samples
   - Creates stratified train/test split
   - Applies bandpass filtering and normalization

2. **Model Loading**
   - Automatically detects and loads fine-tuned model if available
   - Falls back to pre-trained checkpoint if needed
   - Tests multiple configuration attempts for robust loading

3. **Training (if needed)**
   - Stage 1: Train classifier layers only (10 epochs)
   - Stage 2: Fine-tune entire model (optional)
   - Uses learning rate scheduling and early stopping

4. **Evaluation**
   - Classification report with precision/recall/F1
   - Confusion matrix visualization
   - Per-class performance metrics

### Inference on Single File
```python
from train import predict_single_file

# Load model and stats
model = load_model("checkpoints/final_loaded_model.pt")
stats = load_stats()

# Predict
predicted_class = predict_single_file(
    file_path="data/balanced_subset/1_00001.csv",
    true_age=65,
    true_label=1,  # MCI
    model=model,
    stats=stats
)
```

## Training Process

### Stage 1: Classifier Training (10 epochs)
- Freeze backbone (ResNet blocks)
- Train only fully connected layers
- Learning rate: 1e-3
- Batch size: 8
- Class-weighted loss

### Stage 2: Full Fine-tuning (Optional)
- Unfreeze entire model
- Lower learning rate: 1e-4
- Continue training if accuracy < 70%

### Optimization
- **Optimizer**: Adam with weight decay
- **Scheduler**: ReduceLROnPlateau (patience=3, factor=0.5)
- **Loss**: Cross-entropy with class weights
- **Device**: CUDA (GPU) if available

## Results

### Model Performance
- **Test Accuracy**: ~70-80% (depending on fine-tuning)
- **Best Model**: Saved to `checkpoints/best_retrained_model.pt`

### Metrics Tracked
- Training/validation loss
- Training/validation accuracy
- Per-class precision, recall, F1-score
- Confusion matrix

## Key Implementation Details

### Preprocessing Pipeline
```python
1. Load raw CSV → 22 channels × T timepoints
2. Bandpass filter (1-40 Hz) → Remove artifacts
3. Truncate/pad → Fixed length (500)
4. Normalize → Z-score using dataset stats
5. Age normalization → Scale to unit variance
```

### Data Augmentation
Currently not implemented, but potential additions:
- Time warping
- Random cropping
- Gaussian noise
- Amplitude scaling

### Smart Model Loading
The notebook includes intelligent model loading that:
- Attempts to load fine-tuned model first
- Tries multiple architecture configurations
- Falls back to pre-trained checkpoint if needed
- Handles missing keys gracefully
- Verifies model works with test forward pass

## Troubleshooting

### Common Issues

**1. Model Loading Errors**
- Ensure `checkpoints/checkpoint.pt` exists
- Check that model architecture matches saved weights
- The notebook auto-detects configuration from state dict

**2. CUDA Out of Memory**
- Reduce batch size (default: 8)
- Use CPU: `device = torch.device("cpu")`

**3. Poor Performance**
- Check class balance in dataset
- Verify annotation.json is correct
- Increase training epochs
- Adjust learning rate

**4. Data Loading Issues**
- Verify CSV format matches expected structure
- Check annotation.json serial numbers match filenames
- Ensure all required columns are present

## Future Improvements
- [ ] Implement data augmentation
- [ ] Add early stopping with patience
- [ ] Experiment with attention mechanisms
- [ ] Multi-fold cross-validation
- [ ] Ensemble methods
- [ ] Real-time inference pipeline
- [ ] Model interpretability (attention maps, gradient analysis)

## Citations
If you use this code, please cite:
```
@misc{team32_eeg_ad_classification,
  title={EEG-Based Alzheimer's Disease Classification with ResNet1D},
  author={Team 32},
  year={2025},
  publisher={GitHub}
}
```

## License
MIT License

## Contact
For questions or issues, please open an issue on the GitHub repository.
