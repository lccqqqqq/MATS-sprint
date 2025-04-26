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
#%% memory usage
monitor = MemoryMonitor("Diffing Weights")
monitor.start()
monitor.start_continuous_monitoring()

#%%

model_names = {
    "llama-3.1-8b-r1-distilled": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "llama-3.1-8b": "meta-llama/Meta-Llama-3.1-8B",
    "qwen-2.5-7b-math-instruct": "Qwen/Qwen2.5-Math-7B-Instruct",
    "qwen-2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "qwen-2.5-7b-math": "Qwen/Qwen2.5-Math-7B",
    "qwen-2.5-7b": "Qwen/Qwen2.5-7B",
}
models = {}
models["llama-3.1-8b"] = LanguageModel(
    model_names["llama-3.1-8b"], 
    device_map="cpu", 
    dispatch=True,
    torch_dtype=t.bfloat16  # Use bfloat16 precision
)
models["llama-3.1-8b-r1-distilled"] = LanguageModel(
    model_names["llama-3.1-8b-r1-distilled"], 
    device_map="cuda", 
    dispatch=True,
    torch_dtype=t.bfloat16
)
#%% Memory management
models["llama-3.1-8b-r1-distilled"].to("cpu")
models["llama-3.1-8b"].to("cpu")

def relieve_memory():
    t.cuda.empty_cache()
    t.cuda.reset_peak_memory_stats()
    gc.collect()

relieve_memory()

#%% Functions for retrieving attention weights
def retrieve_attn_weights(
    model: LanguageModel,
    augment_kv: bool = True,
    print_shapes: bool = False,
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
            d_head=model.config.head_dim,
        )
        
    attns["O"] = attns["O"].transpose(-2, -1)
    
    if print_shapes:
        for key, value in attns.items():
            print(f"{key}: {value.shape}")

    return attns

# post processing to attention scores

def get_attn_scores(model: LanguageModel):
    attns = retrieve_attn_weights(model)
    Q, K = attns["Q"], attns["K"]
    # broadcast K from n_kv_heads to n_attn_heads
    K_aug = K.repeat_interleave(Q.shape[1] // K.shape[1], dim=1)
    qk = Q @ K_aug.transpose(-2, -1) / math.sqrt(Q.shape[-2]) # normalize by the head dimension
    
    # mask the acausal part
    # scores = t.tril(scores)
    # scores = F.softmax(scores, dim=-1)
    return qk

#%% Get the attention scores for both models
# relieve_memory()
# models["llama-3.1-8b"].to("cuda")
# qk = get_attn_scores(models["llama-3.1-8b"])
# models["llama-3.1-8b"].to("cpu")
# relieve_memory()
# models["llama-3.1-8b-r1-distilled"].to("cuda")
# qk_distilled = get_attn_scores(models["llama-3.1-8b-r1-distilled"])
# models["llama-3.1-8b-r1-distilled"].to("cpu")

# Let's do a more memory-efficient approach, making use of the low-rank property of attention weight matrices.

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
    
    weight_norm = 0.5 * ((
        diag_terms[0]["qq"] @ diag_terms[0]["kk"]
    ).diagonal(dim1=-2, dim2=-1).sum(dim=-1) + (
        diag_terms[1]["qq"] @ diag_terms[1]["kk"]
    ).diagonal(dim1=-2, dim2=-1).sum(dim=-1))
    
    rel_weight_diff = weight_diff / weight_norm
    
    # weight_diff = trace_of_product(
    #     diag_terms[0]["qq"],
    #     diag_terms[0]["kk"]
    # ) + trace_of_product(
    #     diag_terms[1]["qq"],
    #     diag_terms[1]["kk"]
    # ) - trace_of_product(
    #     off_diag_terms["kk'"],
    #     off_diag_terms["q'q"]
    # ) - trace_of_product(
    #     off_diag_terms["k'k"],
    #     off_diag_terms["qq'"]
    # )
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
    weight_norm = 0.5 * ((
        diag_terms[0]["vv"] @ diag_terms[0]["oo"]
    ).diagonal(dim1=-2, dim2=-1).sum(dim=-1) + (
        diag_terms[1]["vv"] @ diag_terms[1]["oo"]
    ).diagonal(dim1=-2, dim2=-1).sum(dim=-1))
    
    # Calculate relative weight difference
    rel_weight_diff = weight_diff / weight_norm
    
    return rel_weight_diff, diag_terms, off_diag_terms

#%% Get the weight differences
models["llama-3.1-8b"].to("cuda")
models["llama-3.1-8b-r1-distilled"].to("cuda")
print("loaded to cuda")
attn_base = retrieve_attn_weights(models["llama-3.1-8b"])
attn_ft = retrieve_attn_weights(models["llama-3.1-8b-r1-distilled"])
models["llama-3.1-8b"].to("cpu")
models["llama-3.1-8b-r1-distilled"].to("cpu")
print("retrieved attn weights, moved models back to cpu")
weight_diff, diag_terms, off_diag_terms = get_attn_weight_diff_ov(
    attn_base,
    attn_ft
)
#%%

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(weight_diff.to(t.float32).detach().cpu().numpy(), cmap='RdBu_r', center=0)
plt.title('Relative Attention Weight Differences')
plt.xlabel('Head')
plt.ylabel('Layer')
# plt.colorbar(label='Weight Difference')

plt.tight_layout()
plt.show()







