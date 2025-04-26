#%% Setup
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

monitor = MemoryMonitor()
monitor.start()
monitor.start_continuous_monitoring()
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)
# %%
model_names = {
    "llama-3.1-8b-r1-distilled": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "llama-3.1-8b": "meta-llama/Meta-Llama-3.1-8B",
    "qwen-2.5-7b-math-instruct": "Qwen/Qwen2.5-Math-7B-Instruct",
    "qwen-2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "qwen-2.5-7b-math": "Qwen/Qwen2.5-Math-7B",
    "qwen-2.5-7b": "Qwen/Qwen2.5-7B",
}

models = {}
for model_name in model_names:
    models[model_name] = LanguageModel(
        model_names[model_name],
        device_map="cpu",
        dispatch=True,
        torch_dtype=t.bfloat16
    )

# %%
def retrieve_attn_weights(
    model: LanguageModel,
    augment_kv: bool = True,
    print_shapes: bool = False,
    model_name: str = "llama-3.1-8b",
):
    attns = {
        "Q": t.stack([model.model.layers[i].self_attn.q_proj.weight for i in range(model.config.num_hidden_layers)]),
        "K": t.stack([model.model.layers[i].self_attn.k_proj.weight for i in range(model.config.num_hidden_layers)]),
        "V": t.stack([model.model.layers[i].self_attn.v_proj.weight for i in range(model.config.num_hidden_layers)]),
        "O": t.stack([model.model.layers[i].self_attn.o_proj.weight for i in range(model.config.num_hidden_layers)]),
    }
    if augment_kv:
        attns["K"] = attns["K"].repeat_interleave(attns["Q"].shape[1] // attns["K"].shape[1], dim=1)
        attns["V"] = attns["V"].repeat_interleave(attns["Q"].shape[1] // attns["V"].shape[1], dim=1)

    for key, value in attns.items():
        attns[key] = einops.rearrange(
            value,
            "n_layers (d_head n_head) d_model -> n_layers n_head d_model d_head",
            d_head=model.config.head_dim if model_name.startswith("llama") else model.config.hidden_size // model.config.num_attention_heads,
        )
        
    attns["O"] = attns["O"].transpose(-2, -1)
    
    if print_shapes:
        for key, value in attns.items():
            print(f"{key}: {value.shape}")

    return attns


def get_attn_weight_diff_qk(
    attn_base: dict[str, t.Tensor],
    attn_ft: dict[str, t.Tensor],
):
    # model_base.to("cuda")
    # model_ft.to("cuda")
    # attn_base = retrieve_attn_weights(model_base)
    # attn_ft = retrieve_attn_weights(model_ft)
    # model_base.to("cpu")
    # model_ft.to("cpu")
    
    diag_terms = [
        {
            "qq": attn["Q"].transpose(-2, -1) @ attn["Q"],
            "kk": attn["K"].transpose(-2, -1) @ attn["K"],
        }
        for attn in [attn_base, attn_ft]
    ]
    off_diag_terms = {
        "kk'": attn_base["K"].transpose(-2, -1) @ attn_ft["K"],
        "q'q": attn_ft["Q"].transpose(-2, -1) @ attn_base["Q"],
        "k'k": attn_ft["K"].transpose(-2, -1) @ attn_base["K"],
        "qq'": attn_base["Q"].transpose(-2, -1) @ attn_ft["Q"],
    }
    
    weight_diff = (
        diag_terms[0]["qq"] @ diag_terms[0]["kk"]
    ).diagonal(dim1=-2, dim2=-1).sum(dim=-1) + (
        diag_terms[1]["qq"] @ diag_terms[1]["kk"]
    ).diagonal(dim1=-2, dim2=-1).sum(dim=-1) - (
        off_diag_terms["kk'"] @ off_diag_terms["q'q"]
    ).diagonal(dim1=-2, dim2=-1).sum(dim=-1) - (
        off_diag_terms["k'k"] @ off_diag_terms["qq'"]
    ).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    
    weight_norm = ((
        diag_terms[0]["qq"] @ diag_terms[0]["kk"]
    ).diagonal(dim1=-2, dim2=-1).sum(dim=-1) + (
        diag_terms[1]["qq"] @ diag_terms[1]["kk"]
    ).diagonal(dim1=-2, dim2=-1).sum(dim=-1))
    
    rel_weight_diff = weight_diff / weight_norm

    return rel_weight_diff, diag_terms, off_diag_terms

def get_attn_weight_diff_ov(
    attn_base: dict[str, t.Tensor],
    attn_ft: dict[str, t.Tensor],
):
    """
    Calculate the relative difference in OV circuit weights between base and fine-tuned models.
    
    Args:
        attn_base: Dictionary containing attention weights for base model
        attn_ft: Dictionary containing attention weights for fine-tuned model
        
    Returns:
        rel_weight_diff: Relative difference in OV circuit weights
        diag_terms: Diagonal terms used in calculation
        off_diag_terms: Off-diagonal terms used in calculation
    """
    # Calculate diagonal terms for both models
    
    # transpose the attn[O] to match the shape of attn[V]
    attn_base["O"] = attn_base["O"].transpose(-2, -1)
    attn_ft["O"] = attn_ft["O"].transpose(-2, -1)
    
    diag_terms = [
        {
            "vv": attn["V"].transpose(-2, -1) @ attn["V"],
            "oo": attn["O"].transpose(-2, -1) @ attn["O"],
        }
        for attn in [attn_base, attn_ft]
    ]
    
    # Calculate off-diagonal terms between models
    off_diag_terms = {
        "vv'": attn_base["V"].transpose(-2, -1) @ attn_ft["V"],
        "o'o": attn_ft["O"].transpose(-2, -1) @ attn_base["O"],
        "v'v": attn_ft["V"].transpose(-2, -1) @ attn_base["V"],
        "oo'": attn_base["O"].transpose(-2, -1) @ attn_ft["O"],
    }
    
    # Calculate weight difference using trace of products
    weight_diff = (
        diag_terms[0]["vv"] @ diag_terms[0]["oo"]
    ).diagonal(dim1=-2, dim2=-1).sum(dim=-1) + (
        diag_terms[1]["vv"] @ diag_terms[1]["oo"]
    ).diagonal(dim1=-2, dim2=-1).sum(dim=-1) - (
        off_diag_terms["vv'"] @ off_diag_terms["o'o"]
    ).diagonal(dim1=-2, dim2=-1).sum(dim=-1) - (
        off_diag_terms["v'v"] @ off_diag_terms["oo'"]
    ).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    
    # Calculate weight norm for normalization
    
    # check the norms
    weight_norm_1 = (
        diag_terms[0]["vv"] @ diag_terms[0]["oo"]
    ).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    weight_norm_2 = (
        diag_terms[1]["vv"] @ diag_terms[1]["oo"]
    ).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    
    weight_norm = weight_norm_1 + weight_norm_2
    
    # Calculate relative weight difference
    rel_weight_diff = weight_diff / weight_norm
    
    return rel_weight_diff, diag_terms, off_diag_terms, weight_norm_1, weight_norm_2

# %% Generic function for comparing model attn weights
def compare_attn_weights(
    model_name_base: str,
    model_name_ft: str,
    attn_type: str = "qk",
    plot_heatmap: bool = True,
    save_heatmap: bool = False,
    save_name: str | None = None,
    save_dir: str = OUT_DIR,
):
    model_base = models[model_name_base]
    model_ft = models[model_name_ft]
    model_base.to("cuda")
    model_ft.to("cuda")
    print("loaded to cuda")
    attn_base = retrieve_attn_weights(model_base, model_name=model_name_base)
    attn_ft = retrieve_attn_weights(model_ft, model_name=model_name_ft)
    model_base.to("cpu")
    model_ft.to("cpu")
    print("retrieved attn weights, moved models back to cpu")

    if attn_type == "qk":
        rel_weight_diff, diag_terms, off_diag_terms = get_attn_weight_diff_qk(attn_base, attn_ft)
    elif attn_type == "ov":
        rel_weight_diff, diag_terms, off_diag_terms, weight_norm_1, weight_norm_2 = get_attn_weight_diff_ov(attn_base, attn_ft)
    else:
        raise ValueError(f"Invalid attn_type: {attn_type}")

    if plot_heatmap:
        # Plot heatmap of relative weight difference
        plt.figure(figsize=(10, 8))
        sns.heatmap(rel_weight_diff.to(t.float32).detach().cpu().numpy(), cmap='RdBu_r', center=0)
        plt.title(
            f"Relative weight diff in {attn_type}: {model_name_base} vs {model_name_ft}"
        )
        plt.xlabel('Head')
        plt.ylabel('Layer')
        plt.tight_layout()
        plt.show()
        
        if save_heatmap:
            if save_name is None:
                raise ValueError("save_name must be provided if save_heatmap is True")
            else:
                plt.savefig(os.path.join(save_dir, save_name))
    
    return rel_weight_diff, diag_terms, off_diag_terms, weight_norm_1, weight_norm_2


# %% Experimenting with qwen

rel_weight_diff, diag_terms, off_diag_terms, weight_norm_1, weight_norm_2 = compare_attn_weights(
    "qwen-2.5-7b",
    "qwen-2.5-7b-math",
    attn_type="ov",
    plot_heatmap=True,
    save_heatmap=True,
    save_name="ov_diff_qwen_base_math.png",
    save_dir=OUT_DIR,
)

# qk_diff_qwen_base_instruct.png: hardly any difference, the negative values are seen as due to lower precision of bfloat16.


# %% Testing the weight difference for qwen
