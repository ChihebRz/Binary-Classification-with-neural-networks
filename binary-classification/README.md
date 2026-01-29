# Binary Classification with Neural Network

## Description

This project focuses on building and evaluating a neural network for binary classification. It processes a given dataset, trains a PyTorch-based neural network model, evaluates its performance, and provides tools for prediction and visualization of decision boundaries.

The dataset used, `column_2C_weka.csv`, contains 6 features related to pelvic and spinal measurements and a 'class' label (Normal/Abnormal), suggesting a task related to identifying spinal disorders.

## Features

*   **Data Preprocessing**: Includes loading, label encoding, feature scaling (StandardScaler), and splitting data into training and testing sets.
*   **Neural Network Model**: A simple feed-forward neural network implemented with PyTorch (`SpineClassifier`).
*   **Model Training**: Training loop with `BCEWithLogitsLoss` and `Adam` optimizer.
*   **Model Evaluation**: Calculates accuracy and displays a confusion matrix on the test set.
*   **Prediction**: Script to make predictions on new data.
*   **Jupyter Notebooks**: Interactive notebooks for data exploration, full pipeline execution, and testing inference.

## Project Structure

```
binary-classification/
├── artifacts/
│   └── model.pth                 # Saved trained PyTorch model state dictionary
├── data/
│   └── column_2C_weka.csv        # Dataset for classification
├── notbooks/
│   ├── a.ipynb                   # Primary notebook for full pipeline execution and exploration
│   ├── exploration.ipynb         # Notebook for initial data exploration
│   └── test_inference.ipynb      # Notebook for testing model inference
├── src/
│   ├── __init__.py               # Makes 'src' a Python package
│   ├── evaluate.py               # Script for model evaluation
│   ├── helper.py                 # Utility functions (e.g., plot_decision_boundary)
│   ├── model.py                  # Neural network architecture definition
│   ├── predict.py                # Script for making predictions
│   ├── preprocess.py             # Script for data loading and preprocessing
│   └── train.py                  # Script for model training
├── .gitignore                    # Git ignore file
└── requirements.txt              # Python dependencies
```

## Installation

To set up the project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd binary-classification
    ```
    (Note: If this is not a Git repository, you can skip this step and ensure you are in the project's root directory.)

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

You can run the different parts of the pipeline using the Python scripts or interactively through the Jupyter notebooks.

### Running Scripts

Navigate to the `src` directory to run the Python scripts:

1.  **Preprocess Data:**
    ```bash
    python src/preprocess.py
    ```
    (This script is mainly for defining the `load_data` function that will be used by `train.py` and `evaluate.py`. Running it directly might not produce output unless explicitly added.)

2.  **Train the Model:**
    ```bash
    python src/train.py
    ```
    This will train the neural network and save the trained model's `state_dict` to `artifacts/model.pth`.

3.  **Evaluate the Model:**
    ```bash
    python src/evaluate.py
    ```
    This will load the trained model, make predictions on the test set, and print accuracy and confusion matrix.

4.  **Make Predictions (Inference):**
    ```bash
    python src/predict.py
    ```
    (You might need to modify `predict.py` to specify input data for inference.)

### Using Jupyter Notebooks

Open `notbooks/a.ipynb` (or `exploration.ipynb`, `test_inference.ipynb`) in your Jupyter environment. The `a.ipynb` notebook provides a consolidated workflow including data loading, preprocessing, model training, evaluation, and visualization of the decision boundary.


## Results

After training and evaluation, the model's performance metrics (accuracy, confusion matrix) will be displayed. The decision boundary plot in `a.ipynb` provides a visual understanding of how the model separates the 'Abnormal' and 'Normal' classes based on the first two principal components of the features.

## Contributing

Feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
