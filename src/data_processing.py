"""
This module handles the loading, and preprocessing of code snippets
from the BugNet dataset.
"""

import re
import logging
import pandas as pd
from typing import Tuple, List, Optional
from datasets import load_dataset
from transformers import RobertaTokenizer
import yaml
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataProcessor:
    
    def __init__(self, config_path: str = "config.yaml"):
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            self.tokenizer = RobertaTokenizer.from_pretrained(
                self.config['model']['pretrained_model']
            )
        except Exception as e:
            logger.error(f"Failed to initialize DataProcessor: {str(e)}")
            raise

    def remove_comments(self, code: str) -> str:
        try:
            code = re.sub(r"#.*", "", code)  # Remove single-line comments
            code = re.sub(r'("""|\'\'\')(.*?)\1', '', code, flags=re.DOTALL)  # Remove multi-line comments
            return code
        except Exception as e:
            logger.error(f"Error removing comments: {str(e)}")
            return code

    def is_valid_python(self, code: str) -> bool:
        try:
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError:
            return False

    def is_reasonable_length(self, code: str) -> bool:
        lines = code.strip().split('\n')
        min_lines = self.config['data']['preprocessing']['min_lines']
        max_lines = self.config['data']['preprocessing']['max_lines']
        return min_lines <= len(lines) <= max_lines

    def is_within_token_range(self, code: str) -> bool:
        try:
            tokens = self.tokenizer(code, truncation=False, return_tensors='pt')
            length = tokens['input_ids'].shape[1]
            min_tokens = self.config['data']['preprocessing']['min_tokens']
            max_tokens = self.config['data']['preprocessing']['max_tokens']
            return min_tokens <= length <= max_tokens
        except Exception as e:
            logger.error(f"Error checking token range: {str(e)}")
            return False

    def load_bugnet_dataset(self) -> pd.DataFrame:
        try:
            logger.info("Loading BugNet dataset...")
            dataset = load_dataset("alexjercan/bugnet", "Python", trust_remote_code=True)
            data = dataset['train']

            buggy_df = pd.DataFrame({'code': data['fail'], 'label': 1})
            clean_df = pd.DataFrame({'code': data['pass'], 'label': 0})
            df = pd.concat([buggy_df, clean_df], ignore_index=True)

            logger.info(f"Initial dataset size: {len(df)}")

            # Apply preprocessing steps
            df['code'] = df['code'].apply(self.remove_comments)
            logger.info(f"After comment removal: {len(df)}")

            df = df[df['code'].apply(self.is_valid_python)]
            logger.info(f"After syntax filtering: {len(df)}")

            df = df[df['code'].apply(self.is_reasonable_length)]
            logger.info(f"After line count filtering: {len(df)}")

            df = df[df['code'].apply(self.is_within_token_range)]
            logger.info(f"After token count filtering: {len(df)}")

            df.drop_duplicates(subset=['code'], inplace=True)
            logger.info(f"After duplicate removal: {len(df)}")

            return df

        except Exception as e:
            logger.error(f"Error loading BugNet dataset: {str(e)}")
            raise

    def save_processed_data(self, df: pd.DataFrame, is_cleaned: bool = True) -> None:
        try:
            # Create directories if they don't exist
            data_type = "cleaned" if is_cleaned else "raw"
            os.makedirs(os.path.dirname(self.config['data'][data_type]['train_path']), exist_ok=True)
            
            # Split and save
            train_df, test_df = self._split_data(df)
            train_df.to_csv(self.config['data'][data_type]['train_path'], index=False)
            test_df.to_csv(self.config['data'][data_type]['test_path'], index=False)
            
            logger.info(f"Saved {data_type} dataset to CSV files")
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            raise

    # training and test data split into 20:80 ratio with fixed random seed of '42'
    def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        from sklearn.model_selection import train_test_split
        return train_test_split(df, test_size=0.2, random_state=42) 