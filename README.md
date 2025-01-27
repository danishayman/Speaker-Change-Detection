# 🎙️ Speaker Change Detection using Deep Learning

A Jupyter notebook implementation of speaker change detection using LSTM-based deep learning models on the IEMOCAP dataset.

## 📋 Overview

This project implements a speaker change detection system using LSTM networks in a Jupyter notebook format. The system processes audio features (MFCC and F0) to identify points in a conversation where speaker transitions occur.

## 🔧 Prerequisites

- Python 3.8+
- Jupyter Notebook/Lab
- TensorFlow 2.x
- librosa
- parselmouth
- numpy
- pandas
- matplotlib
- scikit-learn
- seaborn

## 📦 Setup

1. Clone the repository:
```bash
git clone https://github.com/danishayman/Speaker-Change-Detection.git
cd Speaker-Change-Detection
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the IEMOCAP dataset:
   - The dataset can be obtained from [Kaggle](https://www.kaggle.com/datasets/dejolilandry/iemocapfullrelease/data)
   - Place the downloaded dataset in your working directory

## 📓 Notebook Structure

The project is contained in a single Jupyter notebook with the following sections:

1. **Import Libraries**: Setting up necessary Python packages
2. **Feature Extraction**: 
   - Loading audio files
   - Extracting MFCC and F0 features
   - Defining sliding window parameters
3. **Data Preprocessing**:
   - RTTM parsing
   - Label generation
   - Dataset splitting
4. **Model Development**:
   - Building LSTM model
   - Training with different window sizes
   - Performance evaluation
5. **Results and Analysis**:
   - Visualization of results
   - Confusion matrix analysis
   - Comprehensive performance metrics

## 🚀 Features

- 🎵 Audio feature extraction (MFCC and F0)
- 🪟 Sliding window analysis with various sizes (3, 5, 7, 9 frames)
- 🤖 LSTM-based architecture with batch normalization
- 📊 Comprehensive evaluation metrics and visualizations
- 📈 Experiment analysis with different window sizes

## 💻 Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook speaker_change_detection.ipynb
```

2. Ensure your IEMOCAP dataset path is correctly set in the notebook:
```python
base_path = "path/to/your/IEMOCAP/dataset"
```

3. Run all cells sequentially to:
   - Extract features
   - Process data
   - Train models
   - Visualize results

## 📊 Results

The model's performance across different window sizes:

- Best Window Size: 7 frames
- Peak Accuracy: 66.94%
- Precision: 0.0047
- Recall: 0.6593
- F1-Score: 0.0093

## 🔄 Model Architecture

```python
Sequential([
    Input(shape=input_shape),
    LSTM(128, return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(64),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
```

## 🛠️ Future Improvements

- [ ] Implement data augmentation techniques
- [ ] Explore attention mechanisms
- [ ] Add residual connections
- [ ] Implement curriculum learning
- [ ] Experiment with additional acoustic features
- [ ] Optimize batch size and training epochs with better hardware

## 📚 Citation
```bibtex
@article{busso2008iemocap,
    title     = {IEMOCAP: Interactive emotional dyadic motion capture database},
    author    = {Busso, Carlos and Bulut, Murtaza and Lee, Chi-Chun and 
                 Kazemzadeh, Abe and Mower, Emily and Kim, Samuel and 
                 Chang, Jeannette and Lee, Sungbok and Narayanan, Shrikanth S},
    journal   = {Speech Communication},
    volume    = {50},
    number    = {11},
    pages     = {1150--1162},
    year      = {2008},
    publisher = {Elsevier}
}
```

## ⚠️ Note

The current implementation faces challenges with class imbalance and computational constraints. Future improvements should focus on addressing these limitations for better performance.