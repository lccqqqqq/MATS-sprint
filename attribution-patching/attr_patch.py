# %%
import torch
from nnsight import LanguageModel
from transformers import AutoTokenizer
from datasets import load_dataset
import os
import html
import pandas as pd



device = "cuda"
dtype = torch.bfloat16

# %%
import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'sae-diff', 'src'))
os.chdir("/workspace/sprint")
from data_utils import convert_to_base_tokens, verify_base_tokens
from memory_util import MemoryMonitor
monitor = MemoryMonitor()
monitor.start()

base_model = LanguageModel("meta-llama/Llama-3.1-8B", device_map=device, torch_dtype=dtype)
finetune_model = LanguageModel("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", device_map=device, torch_dtype=dtype)

base_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
finetune_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
monitor.measure("tokenizer")

# %%
messages = [
    # {"role": "user", "content": "What countries border the country whose capital is Antananarivo?"},
    # {"role": "user", "content": "Prove the Riemann Hypothesis."}
    # {"role": "user", "content": "What is 2538 in base 2?"}
    # {"role": "user", "content": "Why is one a prime number?"}
    {"role": "user", "content": "What is the third tallest building in NYC?"}
]
finetune_tokens = finetune_tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")[0]
base_tokens = convert_to_base_tokens(finetune_tokens)

assert verify_base_tokens(base_tokens, base_tokenizer)
monitor.measure("convert_to_base_tokens")
# %%
with finetune_model.generate(finetune_tokens, max_new_tokens=150, do_sample=False, temperature=None, top_p=None) as gen:
    ft_out = finetune_model.generator.output[0].save()

print(finetune_tokenizer.decode(ft_out))
monitor.measure("generate")
# %%
wait_token_idx = 146
print(finetune_tokenizer.decode(ft_out[:wait_token_idx]))
print("Next token:", finetune_tokenizer.decode(ft_out[wait_token_idx]))

prompt_tokens_ft = ft_out[:wait_token_idx]
monitor.measure("prompt_tokens_ft")
# %%
# attribution patching
with finetune_model.trace(prompt_tokens_ft) as tr:
    logits_ft = finetune_model.output[0][0].save()

prompt_tokens_b = convert_to_base_tokens(prompt_tokens_ft)
assert verify_base_tokens(prompt_tokens_b.cpu(), base_tokenizer)

with base_model.trace(prompt_tokens_b) as tr:
    logits_b = base_model.output[0][0].save()

print(logits_b.shape)
print(logits_ft.shape)
monitor.measure("trace")
# %%
from torch.distributions import Categorical
print(logits_ft[-1].argmax().item())
finetune_tokenizer.decode(logits_ft[-1].argmax().item())
monitor.measure("argmax")
# compute the localization of the logits
ft_logit_entropy = Categorical(logits=logits_ft[-1]).entropy()
monitor.measure("entropy")

# %%
import math
print(logits_b[-1].argmax().item())
base_tokenizer.decode(logits_b[-1].argmax().item())
monitor.measure("argmax")
# compute the localization of the logits
base_logit_entropy = Categorical(logits=logits_b[-1]).entropy()
monitor.measure("entropy")
print(f"Finetune logit entropy: {ft_logit_entropy.item()}")
print(f"Base logit entropy: {base_logit_entropy.item()}")
print(f"Maximal logit entropy: {math.log(logits_b[-1].shape[0])}")

# %%
answer_token_indices = torch.tensor([40, 14524], device=device)  # "I", "Wait"

# %%

# better metric than logit diff?

def get_logit_diff(logits, answer_token_indices=answer_token_indices):
    logits = logits[-1, :]
    # correct_logits = logits.gather(0, answer_token_indices[0])
    # incorrect_logits = logits.gather(0, answer_token_indices[1])
    correct_logits = logits[answer_token_indices[0]]
    incorrect_logits = logits[answer_token_indices[1]]
    return correct_logits - incorrect_logits

BASE_BASELINE = get_logit_diff(logits_b)
FT_BASELINE = get_logit_diff(logits_ft)
print(f"Base logit diff: {BASE_BASELINE.item()}")
print(f"Finetune logit diff: {FT_BASELINE.item()}")
monitor.measure("logit_diff")
# %%
def wait_metric(logits, answer_token_indices=answer_token_indices):
    return (get_logit_diff(logits, answer_token_indices) - FT_BASELINE) / (BASE_BASELINE - FT_BASELINE)

print(f"Base Baseline: {wait_metric(logits_b).item()}")
print(f"Finetune Baseline: {wait_metric(logits_ft).item()}")
monitor.measure("wait_metric")
# %%
base_is_clean = False
patch_site = "attn_out"  # "residual", "attn_out", "mlp_out"


if base_is_clean:
    clean_model = base_model
    corrupted_model = finetune_model
    clean_tokens = prompt_tokens_b
    corrupted_tokens = prompt_tokens_ft
else:
    clean_model = finetune_model
    corrupted_model = base_model
    clean_tokens = prompt_tokens_ft
    corrupted_tokens = prompt_tokens_b

monitor.measure("model_setup")

# %%
corrupted_out = []
corrupted_grads = []
with corrupted_model.trace(corrupted_tokens) as tr:
    for l, layer in enumerate(corrupted_model.model.layers):
        if patch_site == "residual":
            layer_out = layer.output[0]
        elif patch_site == "attn_out":
            layer_out = layer.self_attn.output[0]
        elif patch_site == "mlp_out":
            layer_out = layer.mlp.down_proj.output
        corrupted_out.append(layer_out.save().clone().detach())
        corrupted_grads.append(layer_out.grad.save().clone().detach())
    
        monitor.measure(f"trace_{l}")

    logits = corrupted_model.output[0][0].save()
    monitor.measure("logits")
    value = wait_metric(logits).save()
    monitor.measure("wait_metric")
    value.backward()
    monitor.measure("backward")

# print(corrupted_grads)

# %%
clean_out = []
with clean_model.trace(clean_tokens) as tr:
    for l, layer in enumerate(clean_model.model.layers):
        if patch_site == "residual":
            layer_out = layer.output[0]
        elif patch_site == "attn_out":
            layer_out = layer.self_attn.output[0]
        elif patch_site == "mlp_out":
            layer_out = layer.mlp.down_proj.output
        clean_out.append(layer_out.save())

# %%
import einops
patching_results = []

for corrupted_grad, corrupted, clean, layer in zip(
    corrupted_grads, corrupted_out, clean_out, corrupted_model.model.layers
):
    attr = einops.reduce(
        corrupted_grad.value * (clean.value - corrupted.value),
        "batch pos dim -> pos",
        "sum",
    )
    patching_results.append(
        attr.detach().cpu().float().numpy()
    )

# %%
import plotly.express as px
fig = px.imshow(
    patching_results,
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0,
    title=f"Attribution Patching ({'Base into Finetune' if base_is_clean else 'Finetune into Base'}, {patch_site}) over Token Positions",
    labels=dict(x="Token Position", y="Layer", color="Norm. Logit Diff"),
)
fig.show()
# %%
for i, token in enumerate(prompt_tokens_ft):
    print(f"{i}: {finetune_tokenizer.decode(token)}")

# %% Doing activation patching -

# take the same prompt tokens
# These should be different only at the special reserved tokens, namely the second user indicator token
# print(prompt_tokens_ft)
# print(prompt_tokens_b)

# The clean run
# access and save the residue streams
N_LAYERS = finetune_model.model.config.num_hidden_layers
with finetune_model.trace(prompt_tokens_ft) as tracer:
    resid_clean = [
        finetune_model.model.layers[l].output[0].save()
        for l in range(N_LAYERS)
    ]
    monitor.measure("resid_clean")
    # [0] is for removing the batch dimension
    logits_clean = finetune_model.lm_head.output[0].save()
    monitor.measure("logits_clean")
    
    # compute the wait metric
    ft_value = wait_metric(logits_clean).save()
    monitor.measure("wait_metric")

# The corrupted run
with base_model.trace(prompt_tokens_b) as tracer:
    resid_corrupted = [
        base_model.model.layers[l].output[0].save()
        for l in range(N_LAYERS)
    ]
    monitor.measure("resid_corrupted")
    logits_corrupted = base_model.lm_head.output[0].save()
    monitor.measure("logits_corrupted")
    
    # compute the wait metric
    base_value = wait_metric(logits_corrupted).save()
    monitor.measure("wait_metric")

# run with intervention
# Activation Patching Intervention
b2f_patching_results = []
f2b_patching_results = []

from tqdm import tqdm
# Iterate through all the layers
for layer_idx in tqdm(range(N_LAYERS)):
    # b2f_patching_per_layer = []
    f2b_patching_per_layer = []
    # Iterate through all tokens
    for token_idx in range(len(clean_tokens)):
        # # Patching corrupted run at given layer and token
        # # base patched to finetune
        # with base_model.trace(prompt_tokens_b) as tracer:
            
        #     # TODO: In this patching step we have not considered the distribution of activations from the base vs. fine-tuned model. To be refined later.
        #     base_model.model.layers[layer_idx].output[0][:, token_idx, :] = resid_clean[layer_idx][:, token_idx, :]
            
        #     b2f_patched_logits = base_model.lm_head.output[0]
        #     value = wait_metric(b2f_patched_logits).save()
        
        # b2f_patching_per_layer.append(value.item())
        
        with finetune_model.trace(prompt_tokens_ft) as tracer:
            # Apply the patch from the clean hidden states to the corrupted hidden states.
            finetune_model.model.layers[layer_idx].output[0][:, token_idx, :] = resid_corrupted[layer_idx][:, token_idx, :]

            patched_logits = finetune_model.lm_head.output[0]

            f2b_patched_logits = finetune_model.lm_head.output[0]
            value = wait_metric(f2b_patched_logits).save()
        
        f2b_patching_per_layer.append(value.item())

    # b2f_patching_results.append(b2f_patching_per_layer)
    f2b_patching_results.append(f2b_patching_per_layer)
    monitor.measure(f"layer_{layer_idx}")

# %% Visualization

import plotly.express as px
fig = px.imshow(
    b2f_patching_results,
    color_continuous_scale="RdBu",
    color_continuous_midpoint=1,
    title=f"Patching finetune activations into base", # Sorry! variable names are the other way round with the title. Title's correct.
    labels=dict(x="Token Position", y="Layer", color="Norm. Logit Diff"),
)
fig.show()
#%%

fig = px.imshow(
    f2b_patching_results,
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0,
    title=f"Patching base activations into finetune",
)
fig.show()



