# Ensemble Model Evaluation for Image Classification

## Project Description

This project provides a framework to evaluate and compare various ensemble learning techniques for a 3-class image classification problem. The classes are "Healthy," "Unhealthy," and "Rubbish."

The main script, `test_Ground1.py`, takes probability outputs from multiple individual classification models (provided as CSV files), combines them using several different ensemble strategies, and evaluates their performance against a ground truth dataset. It generates detailed performance reports and visualizations.

## Project Structure

-   `test_Ground1.py`: The main executable script that performs data loading, ensembling, evaluation, and visualization.
-   `*.csv`: Input data files containing image names and predicted probabilities from individual models.
    -   `isbi2025-ps3c-test-dataset-annotated.csv`: The ground truth file containing the correct labels for each image.
-   `requirements.txt`: A list of Python dependencies required to run the project.
-   `ensemble_results_comparison.csv`: An output CSV file containing the predictions of each individual model and ensemble method for every image.
-   `*.png`: Output charts and plots visualizing model performance, including:
    -   `model_performance_chart.png`: A bar chart comparing the Accuracy and F1-Score of all models.
    -   `confusion_matrix_stacking.png`: A confusion matrix for the best performing stacking model.
    -   `radar_chart_comparison.png`: A radar chart comparing key metrics for a selection of the best models.
    -   `roc_curve_stacking.png`: ROC curves for the best performing stacking model.

## Setup and Installation

### Dependencies

This project requires Python 3 and the libraries listed in `requirements.txt`.

You can install all dependencies using pip:
```bash
pip install -r requirements.txt
```

## How to Run

To run the full evaluation and generate the output files, execute the main script from the command line:

```bash
python test_Ground1.py
```

The script will print the performance metrics for each model and ensemble to the console and save the output charts and CSV files in the project's root directory.

## Adding a New Model

To add a new model's predictions to the ensemble:
1.  Place your model's prediction CSV file in the root directory. The CSV must contain an `image_name` column and probability columns for the classes (e.g., 'Healthy', 'Unhealthy', 'Rubbish').
2.  Open `test_Ground1.py` and add your model's information to the `files` dictionary, following the existing format:
    ```python
    files = {
        # ... existing models
        "Your_Model_Name": "your_model_file.csv",
    }
    ```
3.  Run the script as described above. Your model will be automatically included in the ensembles and evaluation.
