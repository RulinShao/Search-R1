import re
import os
import datasets
import json
from huggingface_hub import hf_hub_download
from datasets import load_dataset, Dataset

from verl.utils.hdfs_io import copy, makedirs
import argparse


example_answer = "\\(\\boxed{C}\\)"
def make_prefix(dp, template_type):
    question = dp['question']

    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> The correct answer is {example_answer}. </answer>. Question: {question}\n"""
    else:
        raise NotImplementedError
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/gpqa_search')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')

    args = parser.parse_args()

    data_source = 'gpqa'

    # Direct file loading approach
    try:
        # Download the specific JSON files
        main_file = hf_hub_download(
            repo_id="rulins/gpqa_preprocessed",
            filename="main.json",
            repo_type="dataset"
        )
        
        diamond_file = hf_hub_download(
            repo_id="rulins/gpqa_preprocessed",
            filename="diamond.json",
            repo_type="dataset"
        )
        
        # Load JSON data
        with open(main_file, 'r') as f:
            main_data = json.load(f)
        
        with open(diamond_file, 'r') as f:
            diamond_data = json.load(f)
        
        # Convert to datasets
        train_dataset = Dataset.from_list(main_data)
        test_dataset = Dataset.from_list(diamond_data)
        
        print(f"Loaded train dataset with {len(train_dataset)} examples")
        print(f"Loaded test dataset with {len(test_dataset)} examples")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

    # First, transform the data to match the expected format
    def transform_data(examples, split_name, start_idx=0):
        transformed = []
        
        for idx, example in enumerate(examples):
            # Extract the question and remove answer choices from the question
            question_text = example['Question'].strip()
            
            # Get the answer
            # answer = example['Correct Answer']
            answer = example['Correct Choice']
            
            # Create the transformed example with the expected fields
            transformed_example = {
                'id': f"{split_name}_{idx + start_idx}",
                'question': question_text,
                'golden_answers': [answer],
            }
            
            transformed.append(transformed_example)
        
        return Dataset.from_list(transformed)

    # Transform the datasets
    train_dataset = transform_data(train_dataset, 'train')
    test_dataset = transform_data(test_dataset, 'test')

    # Print transformed example for verification
    print("Transformed train example:", train_dataset[0])

    # Add a row to each data item that represents the required format
    def make_map_fn(split):
        def process_fn(example, idx):
            # Make sure question is properly formatted
            example['question'] = example['question'].strip()
                
            # Get the prefix template
            question = make_prefix(example, template_type=args.template_type)
            
            # Create the data dictionary matching NQ format exactly
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "fact-reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "target": example['golden_answers']
                    }
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            
            return data

        return process_fn

    # Apply the transformation
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    # Print final example to verify format
    print("Final processed train example:", train_dataset[0])
    
    # Ensure the datasets are not empty
    assert len(train_dataset) > 0, "Processed train dataset is empty!"
    assert len(test_dataset) > 0, "Processed test dataset is empty!"

    # Create output directory
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    os.makedirs(local_dir, exist_ok=True)

    # Save to parquet format
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    # Verify files were created successfully
    assert os.path.exists(os.path.join(local_dir, 'train.parquet')), "Train parquet file was not created!"
    assert os.path.exists(os.path.join(local_dir, 'test.parquet')), "Test parquet file was not created!"
    
    print(f"Successfully created parquet files in {local_dir}")

    # Copy to HDFS if needed
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
        print(f"Copied parquet files to HDFS: {hdfs_dir}")