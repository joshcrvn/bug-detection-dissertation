# Bug Detection in Python Code using Transformer Models

This project implements an automated bug detection system for Python code using transformer-based models. It compares the performance of models trained on raw and cleaned code samples to analyze the impact of code preprocessing on bug detection accuracy.

# Project Overview

The system uses the BugNet dataset and fine-tunes a CodeBERT model to classify code snippets as either buggy or clean. The project implements two parallel pipelines:

1. **Raw Pipeline**: Trains a model on raw non-preprocessed code
2. **Cleaned Pipeline**: Trains a model on extensively preprocessed code 

# Features

- Comprehensive data preprocessing pipeline
- State-of-the-art transformer model (CodeBERT) for code understanding
- Extensive model evaluation with multiple metrics
- Statistical analysis of preprocessing impact
- Automated report generation
- Experiment tracking with MLflow
- Detailed logging and error handling


# Note: Model weight files (e.g., `pytorch_model.bin`) were excluded from submission to reduce file size. You can retrain the models using `main.py` and the provided datasets.


# Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure the project:
   - Review and modify `config.yaml` as needed
   - Ensure all required directories exist (they will be created automatically)

# Usage

1. Run the complete pipeline:
   ```bash
   python src/main.py
   ```

2. Monitor progress:
   - Check `pipeline.log` for detailed execution logs
   - View MLflow dashboard for experiment tracking:
     ```bash
     mlflow ui
     ```

3. Review results:
   - Check `evaluation/evaluation_report.md` for comprehensive analysis
   - View visualizations in `evaluation/visualizations/`
   - Analyze metrics in `evaluation/metrics/`

# Evaluation Metrics

The system evaluates models using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC
- Precision-Recall AUC

# Visualizations

The following visualizations are generated:
- Confusion matrices
- ROC curves
- Precision-Recall curves
- Metric comparison plots

# License

This project is licensed under the MIT License.

# Acknowledgments

- BugNet dataset creators
- Hugging Face library
- CodeBERT authors 