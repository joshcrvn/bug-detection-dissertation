"""
Script to evaluate the trained bug detection models.
"""

import logging
import yaml
import os
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from evaluation import ModelEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    
    try:
        # Load configuration
        config = load_config()
        logger.info("Loaded configuration")
        
        # Initialize evaluator
        model_evaluator = ModelEvaluator()
        
        # Load models and tokenizers
        logger.info("Loading raw model...")
        raw_model = RobertaForSequenceClassification.from_pretrained(config['model']['output_dirs']['raw'])
        raw_tokenizer = RobertaTokenizer.from_pretrained(config['model']['output_dirs']['raw'])
        
        logger.info("Loading cleaned model...")
        cleaned_model = RobertaForSequenceClassification.from_pretrained(config['model']['output_dirs']['cleaned'])
        cleaned_tokenizer = RobertaTokenizer.from_pretrained(config['model']['output_dirs']['cleaned'])
        
        # Load test data
        raw_test_data = pd.read_csv(config['data']['raw']['test_path'])
        cleaned_test_data = pd.read_csv(config['data']['cleaned']['test_path'])
        
        # Evaluate models
        logger.info("Evaluating raw model...")
        raw_metrics = model_evaluator.evaluate_model(
            raw_model,
            raw_tokenizer,
            raw_test_data,
            is_cleaned=False
        )
        
        logger.info("Evaluating cleaned model...")
        cleaned_metrics = model_evaluator.evaluate_model(
            cleaned_model,
            cleaned_tokenizer,
            cleaned_test_data,
            is_cleaned=True
        )
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        model_evaluator.generate_visualizations(
            raw_metrics['true_labels'],
            raw_metrics['predictions'],
            raw_metrics['probabilities'],
            is_cleaned=False
        )
        model_evaluator.generate_visualizations(
            cleaned_metrics['true_labels'],
            cleaned_metrics['predictions'],
            cleaned_metrics['probabilities'],
            is_cleaned=True
        )
        
        # Perform statistical analysis
        logger.info("Performing statistical analysis...")
        analysis = model_evaluator.perform_statistical_analysis(raw_metrics, cleaned_metrics)
        
        # Generate evaluation report
        logger.info("Generating evaluation report...")
        model_evaluator.generate_evaluation_report(raw_metrics, cleaned_metrics, analysis)
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 