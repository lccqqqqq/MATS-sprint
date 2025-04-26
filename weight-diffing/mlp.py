#%% Setup 
from utils import import_modules_and_load_models
import torch as t
import nnsight
from utils import MemoryMonitor
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

# %% Writing hook functions for nnsight to get the encoder and decoder weights

model_base_name = "llama-3.1-8b"
model_ft_name = "llama-3.1-8b-r1-distilled"

lm = models[model_base_name]
lm_ft = models[model_ft_name]

layer_batch_size = 4


########################################################
# Deprecated, see utils.py for the updated version
# include possible
########################################################
def get_weights(
    lm: LanguageModel,
    lm_ft: LanguageModel,
    act_name: str,
    layers: int | list[int],
):
    """Get the weights from the base and fine-tuned models for a given activation name and layers.
    
    Args:
        lm (LanguageModel): Base language model
        lm_ft (LanguageModel): Fine-tuned language model
        act_name (str): Name of the activation to get weights from, e.g.:
            - "mlp.gate_proj" for MLP gate projection weights
            - "self_attn.q_proj" for attention query projection weights  
            - "self_attn.k_proj" for attention key projection weights
        layers (int | list[int]): Layer number(s) to get weights from
        
    Returns:
        tuple[t.Tensor, t.Tensor]: Tuple containing:
            - weights: Stacked weights from base model
            - weights_ft: Stacked weights from fine-tuned model
    """
    if isinstance(layers, int):
        layers = [layers]
    
    weights = []
    weights_ft = []
    for layer in layers:
        act = lm.model.layers[layer]
        act_ft = lm_ft.model.layers[layer]
        
        for attr in act_name.split("."):
            act = getattr(act, attr)
            act_ft = getattr(act_ft, attr)
        
        weights.append(act.weight.data)
        weights_ft.append(act_ft.weight.data)
    
    weights = t.stack(weights)
    weights_ft = t.stack(weights_ft)
    
    return weights, weights_ft



weights, weights_ft = get_weights(
    lm, lm_ft, "mlp.gate_proj", [0, 1, 2, 3]
)

#%% Get and compare the weights

def compare_weights(
    weights: t.Tensor,
    weights_ft: t.Tensor,
    device: str = "cuda",
    metric: str = "frobenius",
):
    weights.to(device)
    weights_ft.to(device)
    if metric == "frobenius":
        diff = weights - weights_ft
        diff_norm = t.norm(diff, p="fro", dim=(1, 2))
        weights_norm = t.norm(weights, p="fro", dim=(1, 2))
        weights_ft_norm = t.norm(weights_ft, p="fro", dim=(1, 2))
        diff_norm = diff_norm / (weights_norm + weights_ft_norm)
    elif metric == "cossim":
        # use the HS inner product to measure the similarity between matrices
        hs_inner_product = (weights * weights_ft).sum(dim=(1, 2))
        weights_norm = (weights * weights).sum(dim=(1, 2)) ** 0.5
        weights_ft_norm = (weights_ft * weights_ft).sum(dim=(1, 2)) ** 0.5
        
        diff_norm = hs_inner_product / (weights_norm * weights_ft_norm)
    
    # moving back to cpu for memory
    weights.to("cpu")
    weights_ft.to("cpu")
    
    return diff_norm

diff_norm = compare_weights(weights, weights_ft, device="cuda", metric="cossim")
print(diff_norm)

#%% Get the low rank frobenius norm

def get_all_layer_comparison(
    lm: LanguageModel,
    lm_ft: LanguageModel,
    act_name: str,
    metric: str = "frobenius",
):
    # NOTE: potentially need to be batched
    weights, weights_ft = get_weights(
        lm, lm_ft, act_name, list(range(lm.model.config.num_hidden_layers))
    )

    diff_norms = compare_weights(weights, weights_ft, device="cuda", metric=metric)
    return diff_norms

diff_norms = get_all_layer_comparison(lm, lm_ft, "mlp.gate_proj")

#%% Plot the diff norms
from tqdm import tqdm
comps = ["gate_proj", "down_proj", "up_proj"]

diff_norms = dict(key=comps, value=[])
for comp in tqdm(comps):
    diff_norm = get_all_layer_comparison(lm, lm_ft, f"mlp.{comp}")
    diff_norm = diff_norm.to(t.float32).cpu().numpy()
    diff_norms[comp] = diff_norm

#%%
plt.figure(figsize=(10, 6))
plt.plot(diff_norms["gate_proj"], label="gate_proj")
plt.plot(diff_norms["down_proj"], label="down_proj")
plt.plot(diff_norms["up_proj"], label="up_proj")
plt.legend()
plt.title("MLP Weight Difference Norms")
plt.xlabel("Layer")
plt.ylabel(r"$\|W - W'\|_F / ( \|W\|_F + \|W'\|_F )$")
plt.savefig("output/diff_norms_mlp_llama.png", dpi=300)

#%% Try for mlps in qwen

lm = models["qwen-2.5-7b"]
lm_ft = models["qwen-2.5-7b-math"]
diff_norms = dict(key=comps, value=[])
for comp in tqdm(comps):
    diff_norm = get_all_layer_comparison(lm, lm_ft, f"mlp.{comp}")
    diff_norm = diff_norm.to(t.float32).cpu().numpy()
    diff_norms[comp] = diff_norm
#%%
plt.figure(figsize=(10, 6))
plt.plot(diff_norms["gate_proj"], label="gate_proj")
plt.plot(diff_norms["down_proj"], label="down_proj")
plt.plot(diff_norms["up_proj"], label="up_proj")
plt.legend()
plt.title("MLP Weight Difference Norms")
plt.xlabel("Layer")
plt.ylabel(r"$\|W - W'\|_F / ( \|W\|_F + \|W'\|_F )$")
plt.savefig("output/diff_norms_mlp_qwen_base_math.png", dpi=300)
        
#%% function wrapper for plots

def compare_mlp_weights(
    lm_base_name: str,
    lm_ft_name: str,
    plot_line: bool = True,
    save_plot: bool = False,
    plot_name: str = "",
    save_path: str = "output",
    metric: str = "frobenius",
):
    comps = ["gate_proj", "down_proj", "up_proj"]
    lm = models[lm_base_name]
    lm_ft = models[lm_ft_name]
    
    diff_norms = dict(key=comps, value=[])
    for comp in tqdm(comps):
        diff_norm = get_all_layer_comparison(lm, lm_ft, f"mlp.{comp}", metric=metric)
        diff_norm = diff_norm.to(t.float32).cpu().numpy()
        diff_norms[comp] = diff_norm
    
    if plot_line:
        plt.figure(figsize=(10, 6))
        plt.plot(diff_norms["gate_proj"], label="gate_proj")
        plt.plot(diff_norms["down_proj"], label="down_proj")
        plt.plot(diff_norms["up_proj"], label="up_proj")
        plt.legend()
        plt.title(f"MLP Weight Difference Norms for {lm_base_name} and {lm_ft_name}")
        plt.xlabel("Layer")
        plt.ylabel(r"$\|W - W'\|_F / ( \|W\|_F + \|W'\|_F )$" if metric == "frobenius" else r"$\mathrm{Tr}(W^T W') / (\|W\|_F \|W'\|_F)$")
    
    if save_plot:
        plt.savefig(f"{save_path}/diff_{metric}_{plot_name}.png", dpi=300)


# %%
from tqdm import tqdm

batch_processing_models = [
    # ("llama-3.1-8b", "llama-3.1-8b-r1-distilled", "mlp_llama_base_distilled"),
    ("qwen-2.5-7b", "qwen-2.5-7b-math", "mlp_qwen_base_math"),
    ("qwen-2.5-7b", "qwen-2.5-7b-instruct", "mlp_qwen_base_inst"),
    ("qwen-2.5-7b-math", "qwen-2.5-7b-math-instruct", "mlp_qwen_math_mathinst"),
]

for lm_base_name, lm_ft_name, plot_name in tqdm(batch_processing_models):
    compare_mlp_weights(
        lm_base_name=lm_base_name,
        lm_ft_name=lm_ft_name,
        plot_name=plot_name,
        save_path="output/cossim",
        save_plot=True,
        metric="cossim",
    )