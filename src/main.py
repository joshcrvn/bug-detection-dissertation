"""
This script puts the entire project together:
1. Data preprocessing
2. Model training
3. Model evaluation
4. Report generation
"""

import logging
import yaml
import os
from typing import Dict, Any
import pandas as pd
from data_processing import DataProcessor
from model_training import ModelTrainer, BugDetectionDataset
from evaluation import ModelEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

#Load configuration from theYAML file.
def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        raise

#Creates directories for project
def setup_directories(config: Dict[str, Any]) -> None:
    try:
        directories = [
            os.path.dirname(config['data']['raw']['train_path']),
            os.path.dirname(config['data']['raw']['test_path']),
            os.path.dirname(config['data']['cleaned']['train_path']),
            os.path.dirname(config['data']['cleaned']['test_path']),
            config['model']['output_dirs']['raw'],
            config['model']['output_dirs']['cleaned'],
            config['training']['output_dirs']['raw'],
            config['training']['output_dirs']['cleaned'],
            config['training']['logging_dir'],
            config['evaluation']['output_dir'],
            config['evaluation']['visualization_dir']
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
        logger.info("Created all necessary directories")
    except Exception as e:
        logger.error(f"Failed to create directories: {str(e)}")
        raise

def main():
    try:
        # Load configuration
        config = load_config()
        logger.info("Loaded configuration")
        
        # Setup directories
        setup_directories(config)
        
        # Initialize components
        data_processor = DataProcessor()
        model_trainer = ModelTrainer()
        model_evaluator = ModelEvaluator()
        
        # Process data
        logger.info("Processing raw data...")
        raw_df = data_processor.load_bugnet_dataset()
        data_processor.save_processed_data(raw_df, is_cleaned=False)
        
        logger.info("Processing cleaned data...")
        cleaned_df = data_processor.load_bugnet_dataset()
        data_processor.save_processed_data(cleaned_df, is_cleaned=True)
        
        # Train models
        logger.info("Training raw model...")
        raw_model, raw_tokenizer = model_trainer.prepare_model(is_cleaned=False)
        raw_train_dataset = BugDetectionDataset(
            pd.read_csv(config['data']['raw']['train_path']),
            raw_tokenizer
        )
        raw_eval_dataset = BugDetectionDataset(
            pd.read_csv(config['data']['raw']['test_path']),
            raw_tokenizer
        )
        model_trainer.train_model(
            raw_model,
            raw_tokenizer,
            raw_train_dataset,
            raw_eval_dataset,
            is_cleaned=False
        )
        
        logger.info("Training cleaned model...")
        cleaned_model, cleaned_tokenizer = model_trainer.prepare_model(is_cleaned=True)
        cleaned_train_dataset = BugDetectionDataset(
            pd.read_csv(config['data']['cleaned']['train_path']),
            cleaned_tokenizer
        )
        cleaned_eval_dataset = BugDetectionDataset(
            pd.read_csv(config['data']['cleaned']['test_path']),
            cleaned_tokenizer
        )
        model_trainer.train_model(
            cleaned_model,
            cleaned_tokenizer,
            cleaned_train_dataset,
            cleaned_eval_dataset,
            is_cleaned=True
        )
        
        # Evaluate models
        logger.info("Evaluating raw model...")
        raw_metrics = model_evaluator.evaluate_model(
            raw_model,
            raw_tokenizer,
            pd.read_csv(config['data']['raw']['test_path']),
            is_cleaned=False
        )
        
        logger.info("Evaluating cleaned model...")
        cleaned_metrics = model_evaluator.evaluate_model(
            cleaned_model,
            cleaned_tokenizer,
            pd.read_csv(config['data']['cleaned']['test_path']),
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
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 