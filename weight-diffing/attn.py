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

#%% retrieve data with ln

def retrieve_attn_with_ln(
    model: LanguageModel,
    augment_kv: bool = True,
    include_ln: bool = True,
    print_shapes: bool = False,
    model_name: str = "qwen-2.5-7b-math",
):
    """Retrieve the attention weights and the ln weights from the model.
    """
    components = {}
    with t.inference_mode():
        # shapes: (n_layers, n_head, d_model, d_head), the matrix O is transposed
        components["Q"] = t.stack([model.model.layers[i].self_attn.q_proj.weight.data for i in range(model.config.num_hidden_layers)])
        components["K"] = t.stack([model.model.layers[i].self_attn.k_proj.weight.data for i in range(model.config.num_hidden_layers)])
        components["V"] = t.stack([model.model.layers[i].self_attn.v_proj.weight.data for i in range(model.config.num_hidden_layers)])
        components["O"] = t.stack([model.model.layers[i].self_attn.o_proj.weight.data for i in range(model.config.num_hidden_layers)])
        
        if augment_kv:
            components["K"] = components["K"].repeat_interleave(components["Q"].shape[1] // components["K"].shape[1], dim=1)
            components["V"] = components["V"].repeat_interleave(components["Q"].shape[1] // components["V"].shape[1], dim=1)
        
        for key, value in components.items():
            components[key] = einops.rearrange(
                value,
                "n_layers (d_head n_head) d_model -> n_layers n_head d_model d_head",
                d_head=model.config.head_dim if model_name.startswith("llama") else model.config.hidden_size // model.config.num_attention_heads,
            )
        
        components["O"] = components["O"].transpose(-2, -1)
        if include_ln:
            # shapes: (n_layers, d_model)
            components["input_ln"] = t.stack(
                [model.model.layers[i].input_layernorm.weight.data for i in range(model.config.num_hidden_layers)]
            )
            components["post_attn_ln"] = t.stack(
                [model.model.layers[i].post_attention_layernorm.weight.data for i in range(model.config.num_hidden_layers)]
            )
        
        if print_shapes:
            for key, value in components.items():
                print(f"{key}: {value.shape}")

    return components

#%%
def get_attn_diff_qk(
    components_base: dict[str, t.Tensor],
    components_ft: dict[str, t.Tensor],
):
    """
    Get the difference in QK weights between two models.
    """
    comp_list = [components_base, components_ft]
    input_ln_scaling = [
        einops.repeat(
            comp["input_ln"],
            "n_layers d_model -> n_layers n_head d_model d_head",
            n_head=comp["Q"].shape[1],
            d_head=comp["Q"].shape[3],
        )
        for comp in comp_list
    ]
    
    attn_base = {
        "Q": components_base["Q"] * input_ln_scaling[0],
        "K": components_base["K"] * input_ln_scaling[0],
    }
    attn_ft = {
        "Q": components_ft["Q"] * input_ln_scaling[1],
        "K": components_ft["K"] * input_ln_scaling[1],
    }
    
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
    
    weight_norm_base = (
        diag_terms[0]["qq"] @ diag_terms[0]["kk"]
    ).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    weight_norm_ft = (
        diag_terms[1]["qq"] @ diag_terms[1]["kk"]
    ).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    
    weight_norm = weight_norm_base + weight_norm_ft
    
    rel_weight_diff = weight_diff / weight_norm

    return (rel_weight_diff,
            attn_base,
            attn_ft,
            weight_norm_base,
            weight_norm_ft)

def get_attn_diff_ov(
    components_base: dict[str, t.Tensor],
    components_ft: dict[str, t.Tensor],
):
    """
    Get the difference in OV weights between two models.
    
    It is unclear in this case how we should scale according to the layer norm. Hence, at the moment we will simply normalize c.f. Frobenious norm and store the scaling factors as output.
    """

    attn_base = {
        "V": components_base["V"],
        "O": components_base["O"].transpose(-2, -1),
    }
    attn_ft = {
        "V": components_ft["V"],
        "O": components_ft["O"].transpose(-2, -1),
    }
    
    diag_terms = [
        {
            "vv": attn["V"].transpose(-2, -1) @ attn["V"],
            "oo": attn["O"].transpose(-2, -1) @ attn["O"],
        }
        for attn in [attn_base, attn_ft]
    ]
    weight_norm_base = (
        diag_terms[0]["vv"] @ diag_terms[0]["oo"]
    ).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    weight_norm_ft = (
        diag_terms[1]["vv"] @ diag_terms[1]["oo"]
    ).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    
    # Calculate off-diagonal terms between models
    off_diag_terms = {
        "vv'": attn_base["V"].transpose(-2, -1) @ attn_ft["V"],
        "o'o": attn_ft["O"].transpose(-2, -1) @ attn_base["O"],
        "v'v": attn_ft["V"].transpose(-2, -1) @ attn_base["V"],
        "oo'": attn_base["O"].transpose(-2, -1) @ attn_ft["O"],
    }
    
    # Calculate weight difference using trace of products
    rel_weight_diff = (
        diag_terms[0]["vv"] @ diag_terms[0]["oo"]
    ).diagonal(dim1=-2, dim2=-1).sum(dim=-1) / weight_norm_base + (
        diag_terms[1]["vv"] @ diag_terms[1]["oo"]
    ).diagonal(dim1=-2, dim2=-1).sum(dim=-1) / weight_norm_ft - (
        off_diag_terms["vv'"] @ off_diag_terms["o'o"]
    ).diagonal(dim1=-2, dim2=-1).sum(dim=-1) / t.sqrt(weight_norm_base * weight_norm_ft) - (
        off_diag_terms["v'v"] @ off_diag_terms["oo'"]
    ).diagonal(dim1=-2, dim2=-1).sum(dim=-1) / t.sqrt(weight_norm_base * weight_norm_ft)
    
    
    return (rel_weight_diff,
            attn_base,
            attn_ft,
            weight_norm_base,
            weight_norm_ft)

#%% using cos sim instead
def get_attn_sim_qk(
    components_base: dict[str, t.Tensor],
    components_ft: dict[str, t.Tensor],
):
    """
    Get the cosine similarity between Q and K weights between two models.
    """
    comp_list = [components_base, components_ft]
    input_ln_scaling = [
        einops.repeat(
            comp["input_ln"],
            "n_layers d_model -> n_layers n_head d_model d_head",
            n_head=comp["Q"].shape[1],
            d_head=comp["Q"].shape[3],
        )
        for comp in comp_list
    ]
    
    attn_base = {
        "Q": components_base["Q"] * input_ln_scaling[0],
        "K": components_base["K"] * input_ln_scaling[0],
    }
    attn_ft = {
        "Q": components_ft["Q"] * input_ln_scaling[1],
        "K": components_ft["K"] * input_ln_scaling[1],
    }
    
    ktk = attn_ft["K"].transpose(-2, -1) @ attn_base["K"]
    qtq = attn_base["Q"].transpose(-2, -1) @ attn_ft["Q"]
    
    inner_prod = (ktk @ qtq).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    base_norm = (attn_base["K"].transpose(-2, -1) @ attn_base["K"]) @ (attn_base["Q"].transpose(-2, -1) @ attn_base["Q"])
    base_norm = base_norm.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    ft_norm = (attn_ft["K"].transpose(-2, -1) @ attn_ft["K"]) @ (attn_ft["Q"].transpose(-2, -1) @ attn_ft["Q"])
    ft_norm = ft_norm.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    
    sim_qk = inner_prod / t.sqrt(base_norm * ft_norm)
    
    return sim_qk, attn_base, attn_ft, base_norm, ft_norm




#%%  
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
    components_base = retrieve_attn_with_ln(model_base, model_name=model_name_base)
    components_ft = retrieve_attn_with_ln(model_ft, model_name=model_name_ft)
    model_base.to("cpu")
    model_ft.to("cpu")
    print("retrieved attn weights, moved models back to cpu")

    if attn_type == "qk":
        rel_weight_diff, attn_base, attn_ft, weight_norm_base, weight_norm_ft = get_attn_diff_qk(components_base, components_ft)
        # sim_qk, attn_base, attn_ft, base_norm, ft_norm = get_attn_sim_qk(components_base, components_ft)
    elif attn_type == "ov":
        rel_weight_diff, attn_base, attn_ft, weight_norm_base, weight_norm_ft = get_attn_diff_ov(components_base, components_ft)
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
    
    return rel_weight_diff, attn_base, attn_ft, weight_norm_base, weight_norm_ft

#%%

rel_weight_diff, attn_base, attn_ft, weight_norm_base, weight_norm_ft = compare_attn_weights(
    "llama-3.1-8b-r1-distilled",
    "llama-3.1-8b",
    attn_type="ov",
    plot_heatmap=True,
    save_heatmap=True,
    save_name="llama_ov_diff.png",
    save_dir=OUT_DIR,
)

#%%


