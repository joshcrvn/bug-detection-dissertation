"""
This module provides comprehensive evaluation of bug detection models,
including metrics calculation, visualization, and statistical analysis.
"""

import logging
import torch
import yaml
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import mlflow
from mlflow.tracking import MlflowClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Force CPU for evaluation
DEVICE = torch.device("cpu")
logger.info(f"Using device: {DEVICE}")

class ModelEvaluator:
    
    def __init__(self, config_path: str = "config.yaml"):
        
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            # Create output directories
            os.makedirs(self.config['evaluation']['output_dir'], exist_ok=True)
            os.makedirs(self.config['evaluation']['visualization_dir'], exist_ok=True)
            
            # Initialize MLflow
            mlflow.set_tracking_uri("file:./mlruns")
            self.client = MlflowClient()
            
        except Exception as e:
            logger.error(f"Failed to initialize ModelEvaluator: {str(e)}")
            raise

    def evaluate_model(
        self,
        model: torch.nn.Module,
        tokenizer: torch.nn.Module,
        test_data: pd.DataFrame,
        is_cleaned: bool = True
    ) -> Dict[str, float]:
        
        try:
            model = model.to(DEVICE)  # Move model to CPU
            model.eval()
            predictions = []
            true_labels = []
            probabilities = []

            with torch.no_grad():
                for _, row in test_data.iterrows():
                    inputs = tokenizer(
                        row['code'],
                        truncation=True,
                        padding='max_length',
                        max_length=self.config['model']['max_length'],
                        return_tensors='pt'
                    )
                    # Move inputs to CPU
                    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                    
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1)
                    pred = torch.argmax(probs, dim=1).item()
                    
                    predictions.append(pred)
                    true_labels.append(row['label'])
                    probabilities.append(probs[0][1].item())

            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(true_labels, predictions),
                'precision': precision_score(true_labels, predictions),
                'recall': recall_score(true_labels, predictions),
                'f1': f1_score(true_labels, predictions),
                'true_labels': true_labels,
                'predictions': predictions,
                'probabilities': probabilities
            }

            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(true_labels, probabilities)
            metrics['auc_roc'] = auc(fpr, tpr)

            # Calculate precision-recall curve
            precision, recall, _ = precision_recall_curve(true_labels, probabilities)
            metrics['auc_pr'] = auc(recall, precision)

            # Log metrics to MLflow
            model_type = "cleaned" if is_cleaned else "raw"
            with mlflow.start_run(run_name=f"evaluate_{model_type}"):
                # Only log numerical metrics
                mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})

            return metrics

        except Exception as e:
            logger.error(f"Error during model evaluation: {str(e)}")
            raise

    def generate_visualizations(
        self,
        true_labels: List[int],
        predictions: List[int],
        probabilities: List[float],
        is_cleaned: bool = True
    ) -> None:
        
        try:
            model_type = "cleaned" if is_cleaned else "raw"
            viz_dir = self.config['evaluation']['visualization_dir']

            # Confusion Matrix
            cm = confusion_matrix(true_labels, predictions)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {model_type.capitalize()} Model')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(os.path.join(viz_dir, f'confusion_matrix_{model_type}.png'))
            plt.close()

            # ROC Curve
            fpr, tpr, _ = roc_curve(true_labels, probabilities)
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_type.capitalize()} Model')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(viz_dir, f'roc_curve_{model_type}.png'))
            plt.close()

            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(true_labels, probabilities)
            pr_auc = auc(recall, precision)
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {model_type.capitalize()} Model')
            plt.legend(loc="lower left")
            plt.savefig(os.path.join(viz_dir, f'pr_curve_{model_type}.png'))
            plt.close()

            logger.info(f"Visualizations saved to {viz_dir}")

        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            raise

    def perform_statistical_analysis(
        self,
        raw_metrics: Dict[str, float],
        cleaned_metrics: Dict[str, float]
    ) -> Dict[str, float]:
       
        try:
            analysis = {}
            
            # Calculate improvement percentages
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'auc_pr']:
                if metric in raw_metrics and metric in cleaned_metrics:
                    improvement = ((cleaned_metrics[metric] - raw_metrics[metric]) / raw_metrics[metric]) * 100
                    analysis[f'{metric}_improvement'] = improvement

            # Save analysis results
            analysis_df = pd.DataFrame([analysis])
            analysis_df.to_csv(
                os.path.join(self.config['evaluation']['output_dir'], 'statistical_analysis.csv'),
                index=False
            )

            return analysis

        except Exception as e:
            logger.error(f"Error performing statistical analysis: {str(e)}")
            raise

    def generate_evaluation_report(
        self,
        raw_metrics: Dict[str, float],
        cleaned_metrics: Dict[str, float],
        analysis: Dict[str, float]
    ) -> None:
        
        try:
            report_path = os.path.join(self.config['evaluation']['output_dir'], 'evaluation_report.md')
            
            with open(report_path, 'w') as f:
                f.write("# Model Evaluation Report\n\n")
                
                # Model Performance
                f.write("## Model Performance\n\n")
                f.write("### Raw Model\n")
                for metric, value in raw_metrics.items():
                    if isinstance(value, (int, float)):
                        f.write(f"- {metric}: {value:.4f}\n")
                
                f.write("\n### Cleaned Model\n")
                for metric, value in cleaned_metrics.items():
                    if isinstance(value, (int, float)):
                        f.write(f"- {metric}: {value:.4f}\n")
                
                # Statistical Analysis
                f.write("\n## Statistical Analysis\n\n")
                f.write("### Improvement Analysis\n")
                for metric, improvement in analysis.items():
                    f.write(f"- {metric}: {improvement:.2f}%\n")
                
                # Conclusions
                f.write("\n## Conclusions\n\n")
                f.write("1. Impact of Data Cleaning\n")
                f.write("2. Model Performance Comparison\n")
                f.write("3. Recommendations for Future Work\n")
            
            logger.info(f"Evaluation report saved to {report_path}")

        except Exception as e:
            logger.error(f"Error generating evaluation report: {str(e)}")
            raise 