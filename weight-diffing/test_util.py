import unittest
from utils import *
from nnsight import LanguageModel

class TestRetrievals(unittest.TestCase):
    def setUp(self):
        self.model_names = {
            "llama-3.1-8b-r1-distilled": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "llama-3.1-8b": "meta-llama/Meta-Llama-3.1-8B",
            "qwen-2.5-7b-math-instruct": "Qwen/Qwen2.5-Math-7B-Instruct",
            "qwen-2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
            "qwen-2.5-7b-math": "Qwen/Qwen2.5-Math-7B",
            "qwen-2.5-7b": "Qwen/Qwen2.5-7B",
        }

        self.models = {}
        for model_name in self.model_names:
            self.models[model_name] = LanguageModel(
                self.model_names[model_name],
                device_map="cpu",
                dispatch=True,
                torch_dtype=t.bfloat16
            )
        
        pass
    
    def test_get_weights_embeddings(self):
        model_name = "llama-3.1-8b"
        lm = self.models[model_name]
        emb, emb_ft = get_weights(
            lm, lm, "embed_tokens", None
        )
        print(emb.shape, emb_ft.shape)
        pass

if __name__ == "__main__":
    unittest.main()