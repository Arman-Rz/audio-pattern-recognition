# Audio Pattern Recognition using Machine Learning and Deep Learning

## Overview

This project investigates automatic environmental sound classification using the UrbanSound8K dataset.  

Three supervised learning approaches are compared:  

- Support Vector Machine (SVM)
- Random Forest (RF)
- Convolutional Neural Network (CNN)

The project follows a complete machine learning pipeline including audio preprocessing, feature extraction, exploratory analysis, model training, hyperparameter optimization, evaluation, and explainability.  

---

## Dataset

**UrbanSound8K**  

- 10 environmental sound classes
- 8732 labeled audio clips
- Original fold split preserved throughout the experiments  

Dataset:  
https://urbansounddataset.weebly.com/urbansound8k.html  

The dataset is **not included** in this repository because of its size. However, the notebooks download, extract and validate the dataset automatically.  

---

## Project Pipeline

1. Data preparation
2. Feature extraction
3. Exploratory feature analysis
4. Model training and hyperparameter optimization
5. Results analysis and explainability

---

## Repository Structure

```text
audio-pattern-recognition/
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_kmeans_visualization.ipynb
│   ├── 04_model_tuning.ipynb
│   ├── 05_results_analysis_and_explainability.ipynb
│   │
│   └── src/
│       ├── cnn.py
│       ├── data_utils.py
│       ├── evaluation.py
│       ├── feature_extraction.py
│       ├── models.py
│       └── visualization.py
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Models

### Support Vector Machine

- RBF kernel
- Class-balanced training
- Grid search

### Random Forest

- Ensemble of decision trees
- Feature importance analysis
- Grid search

### Convolutional Neural Network

- Log-Mel spectrogram input
- Optuna hyperparameter optimization
- Early stopping
- Grad-CAM explainability

---

## Results

| Model | Accuracy | Macro F1 |
|--------|---------:|---------:|
| SVM | 0.683 | 0.698 |
| Random Forest | 0.665 | 0.680 |
| Tuned CNN | **0.720** | **0.733** |  

The CNN achieved the best overall performance while Random Forest provides interpretable feature importance and SVM serves as a strong classical baseline.

---

## Explainability 

The project includes two explainability techniques:

- Random Forest feature importance
- CNN Grad-CAM visualization

---

## Requirements

Python 3.11+  

**Main Libraries:**  

- numpy
- pandas
- librosa
- scikit-learn
- matplotlib
- seaborn
- torch
- torchvision
- optuna  

**Install:**  

Clone the repository:  
    ```bash
    git clone https://github.com/Arman-Rz/audio-pattern-recognition.git
    ``` 

Install dependencies:  
    ```bash
    pip install -r requirements.txt
    ```

---

## Running the Project

Execute the notebooks in the following order:  

```
01_data_preparation.ipynb  
02_feature_extraction.ipynb
03_kmeans_visualization.ipynb
04_model_tuning.ipynb
05_results_analysis_and_explainability.ipynb
```

**Note:** The notebooks automatically create the required directories, download the UrbanSound8K dataset, preprocess the audio files, extract feaures, train the models, and save the generated results and figures. No manual directory setup is required.  

---

## Licence 

This project is licensed under the MIT Licence. 