# Gender Recognition by Voice using Machine Learning

This project implements a machine learning pipeline to identify gender (male or female) based on acoustic properties of voices and speech. It utilizes simple ML methods to analyze frequency and spectral characteristics of audio signals.

## Project Overview
The core of this project is an end-to-end data science workflow, from data ingestion and preprocessing to model optimization and visualization. The system achieves high accuracy by identifying the most discriminative vocal features, such as the mean fundamental frequency.

### Key Features
* **Using Kaggle 
* **Predictive Modeling:** Implements a Scikit-Learn `DecisionTreeClassifier`.
* **Hyperparameter Tuning:** Uses `GridSearchCV` to optimize tree depth and splitting criteria.
* **Feature Analysis:** Extracts and visualizes feature importance to understand which acoustic properties drive the classification.
* **Model Interpretability:** Generates visual and text-based decision rules for the trained model.

## Dataset Information
The dataset used is the **Gender Recognition by Voice** dataset, which consists of 3,168 recorded voice samples, collected from male and female speakers. 

**Acoustic Features include:**
* `meanfreq`: mean frequency (in kHz)
* `sd`: standard deviation of frequency
* `median`: median frequency (in kHz)
* `IQR`: interquartile range (in kHz)
* `sp.ent`: spectral entropy
* `sfm`: spectral flatness
* `meanfun`: average of fundamental frequency measured across acoustic signal
* `label`: target variable (male or female)

## Getting Started

### Prerequisites
Ensure you have Python installed, along with the following libraries:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn kagglehub
```

### Usage
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YourUsername/gender-recognition-by-voice-ml.git
    cd gender-recognition-by-voice-ml
    ```
2.  **Run the analysis:**
    Open the Jupyter Notebook to step through the training process:
    ```bash
    jupyter notebook Classification_with_Decision_Trees.ipynb
    ```

## Results
* **Accuracy:** The model achieves a classification accuracy of **96.2%** on the test set.
* **Optimized Accuracy:** After hyperparameter tuning (Grid Search), the model maintains strong performance with a cross-validation score of **97.2%**.
* **Top Predictor:** `meanfun` (mean fundamental frequency) was identified as the most critical feature, accounting for over **90%** of the decision-making importance.

### Feature Importance Visualization
The project generates a horizontal bar chart ranking the top 10 features, highlighting the dominance of frequency-based metrics in gender distinction.

## Model Details
* **Algorithm:** Decision Tree Classifier
* **Optimization:** 5-fold Cross-Validation
* **Hyperparameters Tuned:** `max_depth`, `min_samples_split`, `min_samples_leaf`, and `criterion`.
* **Evaluation Metrics:** Precision, Recall, F1-Score, and Accuracy.

## License
This project is for educational purposes. The dataset is sourced from Kaggle under its respective license.

***
