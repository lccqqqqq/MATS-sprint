#%% Setup
import torch as t
from nnsight import LanguageModel
from utils import MemoryMonitor, import_modules_and_load_models, get_weights
from huggingface_hub import list_repo_files
import torch.nn.functional as F
import matplotlib.pyplot as plt
import plotly.express as px
from nnsight import LanguageModel
import einops
import gc
import math
import os
import seaborn as sns
import pandas as pd

monitor, models = import_modules_and_load_models()

# %% Plot the diff 

lm = models["llama-3.1-8b"]
lm_ft = models["llama-3.1-8b-r1-distilled"]

emb, emb_ft = get_weights(
    lm, lm_ft, "embed_tokens", None
)

#%% Compare embeddings 

import torch.nn.functional as F
from tqdm import tqdm
def compare_embeddings(
    lm_base_name: str,
    lm_ft_name: str,
    metric: str = "cossim",
    plot_hist: bool = True,
    save_plot: bool = False,
    save_path: str = "output",
    plot_name: str = "",
):
    emb, emb_ft = get_weights(
        models[lm_base_name], models[lm_ft_name], "embed_tokens", None
    )
    # shape: (n_tokens, d_model)
    if metric == "diff_norm":
        diff = emb - emb_ft
        diff_norm = t.norm(diff, p="fro", dim=1)
        emb_norm = t.norm(emb, p="fro", dim=1)
        emb_ft_norm = t.norm(emb_ft, p="fro", dim=1)
        rel_diff = diff_norm / (emb_norm + emb_ft_norm)
    elif metric == "cossim":
        # use the HS inner product to measure the similarity between matrices
        rel_diff = F.cosine_similarity(emb, emb_ft, dim=1)
    
    if plot_hist:
        fig = px.histogram(
            x=rel_diff.to(t.float32).cpu().numpy(),
            nbins=50,
            title=f"{lm_base_name} and {lm_ft_name} by {metric} difference at embedding",
            labels={'x': f"{metric} difference", 'y': 'Frequency'}
        )
        fig.update_layout(yaxis_type="log")
        fig.show()
        
        if save_plot:
            fig.write_image(f"{save_path}/diff_{metric}_{plot_name}.png")
    
    return rel_diff


batch_processing_models = [
    # ("llama-3.1-8b", "llama-3.1-8b-r1-distilled", "mlp_llama_base_distilled"),
    ("qwen-2.5-7b", "qwen-2.5-7b-math", "emb_qwen_base_math"),
    ("qwen-2.5-7b", "qwen-2.5-7b-instruct", "emb_qwen_base_inst"),
    ("qwen-2.5-7b-math", "qwen-2.5-7b-math-instruct", "emb_qwen_math_mathinst"),
]

for lm_base_name, lm_ft_name, plot_name in tqdm(batch_processing_models):
    cossim = compare_embeddings(
        lm_base_name=lm_base_name,
        lm_ft_name=lm_ft_name,
        metric="cossim",
        plot_hist=True,
        save_plot=True,
        plot_name=plot_name,
        save_path="output/cossim"
    )
    

#%% Testing for relative embedding difference

def compare_relative_embedding_difference(
    lm_base_name: str,
    lm_ft_name: str,
    n_samples: int = 1000,
    n_bins: int = 50,
    plot_hist: bool = True,
    save_plot: bool = False,
    save_path: str = "output/rel_embed",
    plot_name: str = "",
    seed: int | None = None,
):
    if save_plot:
        os.makedirs(save_path, exist_ok=True)
    
    emb, emb_ft = get_weights(
        models[lm_base_name], models[lm_ft_name], "embed_tokens", None
    )
    
    # pick a random token
    if seed is not None:
        t.manual_seed(seed)
    
    random_token = t.randint(0, emb.shape[0], (n_samples, 2))
    
    # compute the embedding differences in base model
    emb_diff_base = (emb[random_token[:, 0]] - emb[random_token[:, 1]]).norm(p="fro", dim=1)
    
    # compute the embedding differences in fine-tuned model
    emb_diff_ft = (emb_ft[random_token[:, 0]] - emb_ft[random_token[:, 1]]).norm(p="fro", dim=1)
    
    # checking whether the *magnitudes* of differences are similar
    norm_diff = t.abs(emb_diff_base - emb_diff_ft) / (t.abs(emb_diff_base) + t.abs(emb_diff_ft) + 1e-5)
    
    if plot_hist:
        fig = px.histogram(
            x=norm_diff.to(t.float32).cpu().numpy(),
            nbins=n_bins,
            title=f"{lm_base_name} and {lm_ft_name} by relative embedding difference",
            labels={'x': 'Difference in embedding magnitudes', 'y': 'Frequency'}
        )
        fig.show()
        
        if save_plot:
            fig.write_image(f"{save_path}/diff_{plot_name}.png")
    
    return norm_diff


#%%

norm_diff = compare_relative_embedding_difference(
    lm_base_name="qwen-2.5-7b",
    lm_ft_name="qwen-2.5-7b-math",
    n_samples=30000,
    n_bins=200,
    plot_hist=True,
    save_plot=True,
    plot_name="qwen_base_math",
    save_path="output/rel_embed",
    seed=4,
)










    
    



        