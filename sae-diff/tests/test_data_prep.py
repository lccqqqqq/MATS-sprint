import unittest
import os
import sys
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_prep import convert_to_base_tokens, new_cached_activation_generator
from transformers import AutoModelForCausalLM, AutoTokenizer
from nnsight import LanguageModel
from datasets import load_dataset
import time
from tqdm import tqdm
import torch
class TestDataPrep(unittest.TestCase):
    def setUp(self):
        self.base_model = LanguageModel("meta-llama/Meta-Llama-3.1-8B", device_map="cuda", dispatch=True, torch_dtype=torch.bfloat16)
        self.base_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
        self.finetune_model = LanguageModel("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", device_map="cuda", dispatch=True, torch_dtype=torch.bfloat16)
        self.finetune_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
        dataset = load_dataset(
            "ServiceNow-AI/R1-Distill-SFT",
            "v1",
            split="train",
            streaming=False,
        )
        self.dataset = dataset.shuffle(seed=42)

    def test_data_generator(self):
        data_generator = new_cached_activation_generator(
            self.base_model,
            self.finetune_model,
            self.base_tokenizer,
            self.finetune_tokenizer,
            self.dataset,
            layer_num=0,
            activation_batch_size=256,
            generator_batch_size=16,
            acts_per_run=100_000,  # Combined parameter (was examples_per_run and max_acts_per_file)
            tokens_per_example=128,
            skip_first_n_tokens=1,
            device="cuda"
        )
        
        t0 = time.time()
        for i in tqdm(range(20)):
            test_dataset = next(data_generator)
            print(test_dataset.shape)
        t1 = time.time()
        print(f"Time taken: {t1 - t0} seconds")
        # expected around 30-40s on RTX 6000 GPU with 48G RAM
        pass


if __name__ == "__main__":
    unittest.main()
    

