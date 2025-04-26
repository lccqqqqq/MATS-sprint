#%% Setup
import torch as t
import nnsight
from utils import MemoryMonitor
from huggingface_hub import list_repo_files
import torch.nn.functional as F
import matplotlib.pyplot as plt
import plotly.express as px
import unicodedata


#%% memory usage
monitor = MemoryMonitor("Diffing Weights")
monitor.start()
monitor.start_continuous_monitoring()

#%% Loading models

model_names = {
    "llama-3.1-8b-r1-distilled": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "llama-3.1-8b": "meta-llama/Meta-Llama-3.1-8B",
    "qwen-2.5-7b-math-instruct": "Qwen/Qwen2.5-Math-7B-Instruct",
    "qwen-2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "qwen-2.5-7b-math": "Qwen/Qwen2.5-Math-7B",
    "qwen-2.5-7b": "Qwen/Qwen2.5-7B",
}



from nnsight import LanguageModel
#%%
models = {}
# Load model with reduced precision (bfloat16)
models["llama-3.1-8b"] = LanguageModel(
    model_names["llama-3.1-8b"], 
    device_map="cuda", 
    dispatch=True,
    torch_dtype=t.bfloat16  # Use bfloat16 precision
)

#%%
# Try loading the distilled model with error handling
# Check available files for the distilled model
from huggingface_hub import list_repo_files
print("Checking available files for DeepSeek model:")
files = list_repo_files(model_names["llama-3.1-8b-r1-distilled"])
print(files)
try:
    models["llama-3.1-8b-r1-distilled"] = LanguageModel(
        model_names["llama-3.1-8b-r1-distilled"], 
        device_map="cuda", 
        dispatch=True,
        torch_dtype=t.bfloat16
    )
except Exception as e:
    print(f"Error loading distilled model: {e}")
    print("This model might require special access or have a different file structure.")

#%% Diffing
import plotly.express as px
import pandas as pd

# accessing embedding weights
weights = [
    models["llama-3.1-8b"].model.embed_tokens.weight,
    models["llama-3.1-8b-r1-distilled"].model.embed_tokens.weight,
]

cos_sim = F.cosine_similarity(weights[0], weights[1], dim=1).to(t.float32).detach().cpu().numpy()
ortho_embeds = cos_sim[cos_sim < 0.1]

fig = px.histogram(
    x=cos_sim,
    nbins=50,
    title='Cosine Similarity Distribution of Embedding Weights',
    labels={'x': 'Cosine Similarity', 'y': 'Frequency'}
)
# Set y-axis to logarithmic scale for better visualization of the distribution
fig.update_layout(yaxis_type="log")

fig.show()



#%% Analyze the orthogonal embeddings: 
# So these are typically special tokens in the model, around 200 of them. Rest of the embeddings are identical.

import unicodedata
import numpy as np
from rich import print as rprint
from rich.table import Table


# Get the indices where cosine similarity is less than 0.1
ortho_indices = np.where(cos_sim < 0.1)[0]

# Get the tokenizers from both models
tokenizer_base = models["llama-3.1-8b"].tokenizer
tokenizer_distilled = models["llama-3.1-8b-r1-distilled"].tokenizer

# Convert indices to tokens using both tokenizers
base_tokens = [tokenizer_base.decode([idx]) for idx in ortho_indices]
distilled_tokens = [tokenizer_distilled.decode([idx]) for idx in ortho_indices]

# def get_unicode_info(text):
#     if not text:  # Handle empty string
#         return "'' (Empty)"
    
#     # If it's a single character, get its name
#     if len(text) == 1:
#         try:
#             name = unicodedata.name(text)
#             return f"{text} ({name})"
#         except ValueError:
#             return f"{text} (Unknown)"
    
#     # For multiple characters, show each character's info
#     char_infos = []
#     for char in text:
#         try:
#             name = unicodedata.name(char)
#             char_infos.append(f"{char} ({name})")
#         except ValueError:
#             char_infos.append(f"{char} (Unknown)")
    
#     return " + ".join(char_infos)

# Print the tokens and their cosine similarities
# print("\nTokens with cosine similarity < 0.1:")
# for i, (base_tok, dist_tok, sim) in enumerate(zip(base_tokens, distilled_tokens, ortho_embeds), 1):
#     # base_info = get_unicode_info(base_tok)
#     # distilled_info = get_unicode_info(dist_tok)
#     print(f"{i}. Base: '{base_tok}' | Distilled: '{dist_tok}' | Similarity: {sim:.3f}")
    
    


def compare_token_embeddings(indices, model_list, model_label_list, msg=""):
    table = Table(title=msg)
    table.add_column("Index", style="cyan")
    for label in model_label_list:
        table.add_column(label, style="magenta")

    table.add_column("Similarity", style="green")
    tokenizers = [model.tokenizer for model in model_list]
    tokens = [
        [tokenizer.decode([idx]) for idx in indices]
        for tokenizer in tokenizers
    ]
    for i, idx in enumerate(indices):
        table.add_row(
            f"{i}",
            *[f"{tokens[i_model][i]}" for i_model in range(len(model_list))],
            f"{cos_sim[idx]:.3f}"
        )
    rprint(table)
    # TODO: create a table using rich
    pass
            
compare_token_embeddings(ortho_indices, [models["llama-3.1-8b"], models["llama-3.1-8b-r1-distilled"]], ["Base", "Distilled"])        


#%% Analyze the nearly aligned embeddings: nothing much to see here
aligned_embeds = cos_sim[cos_sim > 0.95]
aligned_indices = np.where(cos_sim > 0.95)[0]
perm = np.argsort(aligned_embeds)[0:200]
fig = px.histogram(
    x=aligned_embeds,
    nbins=20,
    title='Cosine Similarity Distribution of Embedding Weights',
    labels={'x': 'Cosine Similarity', 'y': 'Frequency'}
)
fig.show()


compare_token_embeddings(aligned_indices[perm], [models["llama-3.1-8b"], models["llama-3.1-8b-r1-distilled"]], ["Base", "Distilled"])

#%% Now into the attention heads

Q = models["llama-3.1-8b"].model.layers[0].self_attn.q_proj.weight
K = models["llama-3.1-8b"].model.layers[0].self_attn.k_proj.weight
V = models["llama-3.1-8b"].model.layers[0].self_attn.v_proj.weight
O = models["llama-3.1-8b"].model.layers[0].self_attn.o_proj.weight

#%% Access the model configuration hyperparameters

import einops
import math
def retrieve_attn_weights(
    model: LanguageModel,
):
    attns = {
        "Q": t.stack([model.model.layers[i].self_attn.q_proj.weight for i in range(model.config.num_hidden_layers)]),
        "K": t.stack([model.model.layers[i].self_attn.k_proj.weight for i in range(model.config.num_hidden_layers)]),
        "V": t.stack([model.model.layers[i].self_attn.v_proj.weight for i in range(model.config.num_hidden_layers)]),
        "O": t.stack([model.model.layers[i].self_attn.o_proj.weight for i in range(model.config.num_hidden_layers)]),
    }
    for key, value in attns.items():
        attns[key] = einops.rearrange(
            value,
            "n_layers (d_head n_head) d_model -> n_layers n_head d_model d_head",
            d_head=model.config.head_dim,
        )
    return attns

# post processing to attention scores

def get_attn_scores(attns: dict[str, t.Tensor]):
    Q, K, V, O = attns["Q"], attns["K"], attns["V"], attns["O"]
    # broadcast K from n_kv_heads to n_attn_heads
    K_aug = K.repeat_interleave(attns["Q"].shape[1] // attns["K"].shape[1], dim=1)
    qk = Q @ K_aug.transpose(-2, -1) / math.sqrt(attns["Q"].shape[-2]) # normalize by the head dimension
    
    # mask the acausal part
    # scores = t.tril(scores)
    # scores = F.softmax(scores, dim=-1)
    return qk

# attns = retrieve_attn_weights(models["llama-3.1-8b"])
# qk = get_attn_scores(attns)

#%% Compare the W_QK for both models


lm = models["llama-3.1-8b"]
lm_distilled = models["llama-3.1-8b-r1-distilled"]

# include memory managements
lm.to("cuda")
lm_distilled.to("cpu")
lm_attn = retrieve_attn_weights(lm)
qk = get_attn_scores(lm_attn)

lm.to("cpu")
lm_distilled.to("cuda")
lm_distilled_attn = retrieve_attn_weights(lm_distilled)
qk_distilled = get_attn_scores(lm_distilled_attn)



#%%
# visualize the similarities between the qk matrix elements
diff_frob_norm = t.norm(qk - qk_distilled, p="fro", dim=(-2, -1)).mean(dim=1)
# shape (n_layers,)

fig = px.line(
    diff_frob_norm.to(t.float32).detach().cpu().numpy(),
    title='Difference in Frobenius Norm of QK Matrices',
    labels={'x': 'Layer', 'y': 'Difference in Frobenius Norm'}
)
fig.show()

