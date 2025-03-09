# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the gpqa dataset to parquet format
"""

import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse


def make_prefix(dp, template_type):
    question = dp['question']

    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
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
        # Try loading the repository as a whole first
        dataset = datasets.load_dataset('rulins/gpqa_preprocessed')
        
        # Check what splits are available
        print(f"Available splits: {list(dataset.keys())}")
        
        # If the structure isn't as expected, load files directly
        from huggingface_hub import hf_hub_download
        from datasets import load_dataset, Dataset
        import json
        import os
        
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
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            # The data is already preprocessed, so we can use the fields directly
            question = example['Question']
            correct_choice = example['Correct Choice']
            
            # For golden answers, we'll use the 'Correct Answer' field directly
            golden_answers = [example['Correct Answer']]
            
            solution = {
                "target": correct_choice,
            }

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "fact-reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'id': example['id'],
                    'domain': example['High-level domain'],
                    'subdomain': example['Subdomain']
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
