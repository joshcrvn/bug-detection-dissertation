"""
This module handles the training and fine-tuning of transformer-based models
for bug detection using the processed BugNet dataset.
"""

import logging
import torch
import yaml
import os
from typing import Dict, Any, Optional, Tuple
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments
)
from torch.utils.data import Dataset
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Device selection: forced on CPU for maximum reliability
DEVICE = torch.device("cpu")
logger.info(f"Using device: {DEVICE}")

# tokenisation and prep for code samples
class BugDetectionDataset(Dataset):
    
    def __init__(self, dataframe: pd.DataFrame, tokenizer: RobertaTokenizer, max_length: int = 512):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    #returns number of samples in the dataset
    def __len__(self) -> int:
        return len(self.data)

    #tokenises the inputs from the dataset
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[index]
        encoding = self.tokenizer(
            row['code'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        label = torch.tensor(row['label'], dtype=torch.long)
        return {**{k: v.squeeze(0) for k, v in encoding.items()}, 'labels': label}

class ModelTrainer:
    def __init__(self, config_path: str = "config.yaml"):
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            # Initialize MLflow
            mlflow.set_tracking_uri("file:./mlruns")
            self.client = MlflowClient()
            
            # Create experiment if it doesn't exist
            experiment_name = "bug_detection"
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
            
        except Exception as e:
            logger.error(f"Failed to initialize ModelTrainer: {str(e)}")
            raise

    def prepare_model(self, is_cleaned: bool = True) -> Tuple[RobertaForSequenceClassification, RobertaTokenizer]:
        try:
            model_name = self.config['model']['pretrained_model']
            tokenizer = RobertaTokenizer.from_pretrained(model_name)
            model = RobertaForSequenceClassification.from_pretrained(
                model_name,
                num_labels=self.config['model']['num_labels']
            )
            model = model.to(DEVICE)  # Moves model to selected device (CPU)
            return model, tokenizer
        except Exception as e:
            logger.error(f"Error preparing model: {str(e)}")
            raise

    def train_model(
        self,
        model: RobertaForSequenceClassification,
        tokenizer: RobertaTokenizer,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        is_cleaned: bool = True,
        resume_from_checkpoint: bool = False
    ) -> None:
       
        try:
            # Set up training arguments
            model_type = "cleaned" if is_cleaned else "raw"
            output_dir = self.config['training']['output_dirs'][model_type]
            
            # Check for existing checkpoints if resuming
            latest_checkpoint = None
            if resume_from_checkpoint:
                latest_checkpoint = self._get_latest_checkpoint(output_dir)
                if latest_checkpoint is not None:
                    logger.info(f"Resuming from checkpoint: {latest_checkpoint}")
                else:
                    logger.info("No checkpoint found, starting from scratch")
            
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=self.config['training']['epochs'],
                per_device_train_batch_size=self.config['training']['batch_size'],
                per_device_eval_batch_size=self.config['training']['batch_size'],
                warmup_steps=self.config['training']['warmup_steps'],
                weight_decay=self.config['training']['weight_decay'],
                logging_dir=os.path.join(output_dir, 'logs'),
                logging_steps=self.config['training']['logging_steps'],
                save_strategy='steps',
                save_steps=100,
                save_total_limit=3,
                evaluation_strategy='steps',
                eval_steps=100,
                load_best_model_at_end=True,
                metric_for_best_model='eval_loss',
                greater_is_better=False,
                seed=self.config['training']['seed']
            )

            # Initialize trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
            )

            # Start MLflow run
            with mlflow.start_run(run_name=f"train_{model_type}"):
                # Log parameters (exclude dictionary-type params)
                training_params = {k: v for k, v in self.config['training'].items() if not isinstance(v, dict)}
                model_params = {k: v for k, v in self.config['model'].items() if not isinstance(v, dict)}
                mlflow.log_params(training_params)
                mlflow.log_params(model_params)

                # Train model
                logger.info(f"Starting training for {model_type} model...")
                trainer.train(resume_from_checkpoint=latest_checkpoint)

                # Log metrics
                metrics = trainer.evaluate()
                mlflow.log_metrics(metrics)

                # Save model
                model_path = self.config['model']['output_dirs'][model_type]
                trainer.save_model(model_path)
                tokenizer.save_pretrained(model_path)
                mlflow.log_artifacts(model_path)

                logger.info(f"Training complete for {model_type} model. Model saved to {model_path}")

        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise

        # Output the path to the latest checkpoint 
    def _get_latest_checkpoint(self, output_dir: str) -> Optional[str]:
        
        try:
            # List all checkpoint directories
            checkpoint_dirs = [d for d in os.listdir(output_dir) if d.startswith('checkpoint-')]
            if not checkpoint_dirs:
                return None
            
            # Sort by checkpoint number and get the latest
            latest_checkpoint = sorted(checkpoint_dirs, key=lambda x: int(x.split('-')[1]))[-1]
            return os.path.join(output_dir, latest_checkpoint)
        except Exception as e:
            logger.warning(f"Error finding latest checkpoint: {str(e)}")
            return None

    # loads the trained model and the tokeniser
    def load_trained_model(self, is_cleaned: bool = True) -> Tuple[RobertaForSequenceClassification, RobertaTokenizer]:
       
        try:
            model_type = "cleaned" if is_cleaned else "raw"
            model_path = self.config['model']['output_dirs'][model_type]
            
            model = RobertaForSequenceClassification.from_pretrained(model_path)
            tokenizer = RobertaTokenizer.from_pretrained(model_path)
            
            return model, tokenizer
        except Exception as e:
            logger.error(f"Error loading trained model: {str(e)}")
            raise 


