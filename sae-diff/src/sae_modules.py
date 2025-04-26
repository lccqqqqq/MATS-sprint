# Implementation of Gated SAE
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import einops
from torch import Tensor
from typing import Optional, Tuple, Dict, Any, Generator
from nnsight import LanguageModel
from jaxtyping import Float
from dataclasses import dataclass
from torch.distributions import Categorical
from einops import einsum, rearrange


@dataclass
class GatedSAEConfig:
    sparsity_coeff: float = 0.01
    aux_coeff: float = 0.01
    d_model: int = 768
    d_sae: int = 1024
    
    # other bookkeeping
    weight_normalize_eps: float = 1e-6
    standardize_method: str = "per_token"
    # accepting "plain", "per_token", "per_batch"
    


class GatedDiffSAE(nn.Module):
    def __init__(self, cfg: GatedSAEConfig, device: t.device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        # initialize the parameters
        # By default did not include options for weight-tying
        self.W_dec = nn.Parameter(t.empty(cfg.d_sae, cfg.d_model))
        self.b_dec = nn.Parameter(t.zeros(cfg.d_model))
        self.W_gate = nn.Parameter(t.empty(cfg.d_model, cfg.d_sae))
        self.b_gate = nn.Parameter(t.zeros(cfg.d_sae))
        self.r_mag = nn.Parameter(t.zeros(cfg.d_sae))
        self.b_mag = nn.Parameter(t.zeros(cfg.d_sae))
        
        self._init_weights()
        self.to(self.device)
        
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_gate)
        nn.init.kaiming_uniform_(self.W_dec)
        # the biases are already initialized to zero
        # implement normalization of decoder weights
        with t.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    @property
    def W_mag(self) -> Float[Tensor, "d_model d_sae"]:
        return self.r_mag.exp().unsqueeze(0) * self.W_gate
    

    def forward(
        self,
        x: Float[Tensor, "batch 2d_model"],
    ) -> tuple[
        dict[str, Float[Tensor, "batch"]],
        Float[Tensor, "batch"],
        Float[Tensor, "batch d_sae"],
        Float[Tensor, "batch d_model"],
        Float[Tensor, "batch d_model"],
    ]:
        """
        Implement the forward pass with the gated SAE.
        """
        src = x[:, :self.cfg.d_model] # the pass only acts on the source
        tgt = x[:, self.cfg.d_model:]
        
        
        # implement post processing
        if self.cfg.standardize_method == "plain":
            # no normalization/centering
            diff = tgt - src
            diff_stdized = diff - self.b_dec
            scale_cache = None
        elif self.cfg.standardize_method == "per_token":
            diff = tgt - src
            mu = diff.mean(0)
            std = diff.std(0)
            diff_centered = (diff - mu) / (std + self.cfg.weight_normalize_eps)
            diff_stdized = diff_centered - self.b_dec
            scale_cache = {
                "mu": mu,
                "std": std,
            }
        elif self.cfg.standardize_method == "per_batch":
            diff = tgt - src
            mu = diff.mean(0) 
            diff_centered_batch = diff - mu # such that upon centering, summing over all entries are 0, and this is done via subtracting one global vector of shape (d_model,)
            norm_scale = diff_centered_batch.norm(dim=1).mean() # this is to ensure that the norm the centered diff is 1 across the dataset, instead of individually.
            diff_centered = diff_centered_batch / (norm_scale + self.cfg.weight_normalize_eps)
            diff_stdized = diff_centered - self.b_dec
            scale_cache = {
                "mu": mu,
                "norm_scale": norm_scale,
            }
        elif self.cfg.standardize_method == "new_norm":
            # First do batch norm on the different activations
            src_bn = src / (src.norm() + self.cfg.weight_normalize_eps)
        else:
            raise NotImplementedError(f"Invalid standardization method: {self.cfg.standardize_method}")
        

        # Compute the gating terms (pi_gate(x) and f_gate(x) in the paper)
        gating_pre_activation = (
            einops.einsum(diff_stdized, self.W_gate, "batch d_in, d_in d_sae -> batch d_sae") + self.b_gate
        )
        active_features = (gating_pre_activation > 0).float()

        # Compute the magnitude term (f_mag(x) in the paper)
        magnitude_pre_activation = (
            einops.einsum(diff_stdized, self.W_mag, "batch d_in, d_in d_sae -> batch d_sae") + self.b_mag
        )
        feature_magnitudes = F.relu(magnitude_pre_activation)

        # Compute the hidden activations (f˜(x) in the paper)
        acts_post = active_features * feature_magnitudes

        # Compute reconstructed input
        diff_reconstructed = (
            einops.einsum(acts_post, self.W_dec, "batch d_sae, d_sae d_in -> batch d_in") + self.b_dec
        )

        # Compute loss terms
        gating_post_activation = F.relu(gating_pre_activation)
        via_gate_reconstruction = (
            einops.einsum(
                gating_post_activation, self.W_dec.detach(), "batch d_sae, d_sae d_in -> batch d_in"
            )
            + self.b_dec.detach()
        )
        
        
        loss_dict = {
            "L_reconstruction": (diff_reconstructed - diff_stdized).pow(2).mean(-1),
            "L_sparsity": gating_post_activation.sum(-1),
            "L_aux": (via_gate_reconstruction - diff_stdized).pow(2).sum(-1),
        }

        loss = loss_dict["L_reconstruction"] + self.cfg.sparsity_coeff * loss_dict["L_sparsity"] + self.cfg.aux_coeff * loss_dict["L_aux"]

        assert sorted(loss_dict.keys()) == ["L_aux", "L_reconstruction", "L_sparsity"]
        return loss_dict, loss, acts_post, diff_stdized, diff_reconstructed, scale_cache
    
    @t.no_grad()
    def resampling(
        self,
        batch_data: Float[Tensor, "batch 2d_model"], 
        frac_active_in_window: Float[Tensor, "window d_sae"], # this intends to be in the model keeping track of a mask signalling whether a feature is active
        resample_scale: float,
    ) -> None:
        # generate next batch of data with post processing
        # Directly use the batch data without generating new ones. Should be equivalent and more memory efficient...
        # batch_data = next(resampling_generator) # (batch, d_model * 2)
        loss_dict, _, _, diff, _, _ = self.forward(batch_data)
        l2_loss = loss_dict["L_reconstruction"]
        # fraction of active features in the window
        is_dead = (frac_active_in_window < 1e-8).all(dim=0)
        dead_latents = t.nonzero(is_dead).squeeze(-1)
        n_dead = dead_latents.numel()
        if n_dead == 0:
            return  # If we have no dead features, then we don't need to resample

        if l2_loss.max() < 1e-6:
            return

        # Draw `d_sae` samples from [0, 1, ..., batch_size-1], with probabilities proportional to l2_loss
        distn = Categorical(probs=l2_loss.pow(2) / l2_loss.pow(2).sum())
        replacement_indices = distn.sample((n_dead,))  # type: ignore

        # Index into the batch of hidden activations to get our replacement values
        replacement_values = (diff - self.b_dec)[replacement_indices]   # [n_dead d_in]
        replacement_values_normalized = replacement_values / (
            replacement_values.norm(dim=-1, keepdim=True) + self.cfg.weight_normalize_eps
        )
        
        # --- replace decoder weights for dead latents ---
        self.W_dec.data[dead_latents, :] = replacement_values_normalized
        
        # --- replace value-path encoder weights for dead latents ---
        val_norm_alive_mean = self.W_mag[:, ~is_dead].norm(dim=0).mean().item() if (~is_dead).any() else 1.0
        self.W_mag.data[:, dead_latents] = replacement_values_normalized.T * val_norm_alive_mean * resample_scale

        # --- replace gated-path encoder weights for dead latents ---
        gate_norm_alive_mean = self.W_gate[:, ~is_dead].norm(dim=0).mean().item() if (~is_dead).any() else 1.0
        self.W_gate.data[:, dead_latents] = replacement_values_normalized.T * gate_norm_alive_mean * resample_scale
        
        # --- replace the biases for dead latents ---
        self.b_gate.data[dead_latents] = -1.0
        self.b_mag.data[dead_latents] = 0.0


"""
Dimension key:

B: batch_size
D: d_model (model dimension)
F: d_sae (feature/dictionary size)
Z: d_in (hidden dimension - double model dimension for crosscoder: 2*D)
"""

@dataclass
class BatchTopKSAEConfig:
    d_model: int
    d_sae: int
    k: int
    device: t.device
    standardize_method: str
    weight_normalize_eps: float
    aux_coeff: float


class BatchTopKSAE(nn.Module):
    def __init__(
        self, 
        cfg: BatchTopKSAEConfig,
    ):
        super().__init__()
        
        self.d_model = cfg.d_model
        self.d_sae = cfg.d_sae
        self.k = cfg.k
        self.W_enc = nn.Parameter(t.empty(cfg.d_model, cfg.d_sae))
        self.b_enc = nn.Parameter(t.zeros(cfg.d_sae))
        self.W_dec = nn.Parameter(t.empty(cfg.d_sae, cfg.d_model))
        self.b_dec = nn.Parameter(t.zeros(cfg.d_model))
        
        self.latent_tracker = LatentTracker(
            n_latents=cfg.d_sae,
            dead_threshold=10_000_000,
            device=cfg.device,
        )
        self.standardize_method = cfg.standardize_method
        self.weight_normalize_eps = cfg.weight_normalize_eps
        self.aux_coeff = cfg.aux_coeff
        
        self._init_weights()
        self.device = cfg.device
        self.to(cfg.device)
        
    def _init_weights(self):
        initial_weight_DF = t.empty(self.d_model, self.d_sae)
        nn.init.kaiming_uniform_(initial_weight_DF)
        
        with t.no_grad():
            # Initialize encoder weight: stack the same initial weights for both parts
            # of the input (Z = 2*D)
            self.W_enc.data = initial_weight_DF.clone()
            # Initialize decoder weight as transpose of encoder
            self.W_dec.data = self.W_enc.T.data.clone()
    
    def get_latent_activations(self, diff: t.Tensor) -> t.Tensor:
        activations_BF = einsum(diff, self.W_enc, "b z, z f -> b f") + self.b_enc
        return F.relu(activations_BF)
    
    def auxiliary_loss(self, dead_latents, error, kaux=512):
        """Calculate auxiliary loss using dead latents"""
        if not dead_latents.any():
            return t.tensor(0.0, device=error.device)
        
        # Get pre-activations for dead latents only
        with t.no_grad():
            pre_acts = self.get_latent_activations(error)  # Get all pre-activations
            values = pre_acts * self.W_dec.norm(dim=1)
            dead_values = values[:, dead_latents]  # Select only dead latents
            dead_values = F.relu(dead_values)
        
        # Get top kaux dead latents
        k = min(kaux, dead_values.shape[1])
        top_k_values, top_k_indices = t.topk(dead_values, k, dim=1)
        threshold = top_k_values[..., -1, None]
        mask = dead_values >= threshold
        
        # Only reconstruct using masked dead pre-activations
        masked_features = t.zeros_like(pre_acts)
        masked_features[:, dead_latents] = pre_acts[:, dead_latents] * mask
        
        # Reconstruct error using dead latents
        error_reconstruction = self.decode(masked_features)
        # Calculate MSE
        error_mse = F.mse_loss(error_reconstruction, error)
        
        return error_mse
    
    def apply_batchtopk(self, activations_BF: t.Tensor) -> t.Tensor:
        # activations_BF: [batch_size, d_sae]
        batch_size = activations_BF.shape[0]
        
        decoder_norms_F = t.norm(self.W_dec, dim=1)
        
        # Calculate value scores
        value_scores_BF = einsum(activations_BF, decoder_norms_F, "b f, f -> b f")
        
        # Flatten and find top-k
        flat_scores_bF = rearrange(value_scores_BF, "b f -> (b f)")
        
        # Find top k*batch_size activations
        topk_indices = t.topk(flat_scores_bF, k=self.k * batch_size, dim=0).indices
        
        # Create sparse mask and apply
        mask_bF = t.zeros_like(flat_scores_bF)
        mask_bF[topk_indices] = 1.0
        mask_BF = rearrange(mask_bF, "(b f) -> b f", b=batch_size)
        
        # Apply mask to get sparse activations
        sparse_activations_BF = activations_BF * mask_BF
        
        return sparse_activations_BF
    
    def decode(self, sparse_activations_BF: t.Tensor) -> t.Tensor:
        return einsum(sparse_activations_BF, self.W_dec, "b f, f d -> b d") + self.b_dec
    
    def forward(self, x: t.Tensor) -> tuple[dict[str, t.Tensor], t.Tensor, t.Tensor, t.Tensor, t.Tensor, dict[str, t.Tensor]]:
        src = x[:, :self.d_model]
        tgt = x[:, self.d_model:]
        
        if self.standardize_method == "layer_norm":
            src_mean = src.mean(dim=1)
            src_var = (src - src_mean).pow(2).mean(dim=1)
            src_ln = (src - src_mean) / t.sqrt(src_var + self.weight_normalize_eps)
            tgt_mean = tgt.mean(dim=1)
            tgt_var = (tgt - tgt_mean).pow(2).mean(dim=1)
            tgt_ln = (tgt - tgt_mean) / t.sqrt(tgt_var + self.weight_normalize_eps)
            diff = tgt_ln - src_ln
            scale_cache = {
                "src_mean": src_mean,
                "src_var": src_var,
                "tgt_mean": tgt_mean,
                "tgt_var": tgt_var,
            }
        elif self.standardize_method == "layer_norm_and_diff_layer_norm":
            src_mean = src.mean(dim=1, keepdim=True)
            src_var = (src - src_mean).pow(2).mean(dim=1, keepdim=True)
            src_ln = (src - src_mean) / t.sqrt(src_var + self.weight_normalize_eps)
            tgt_mean = tgt.mean(dim=1, keepdim=True)
            tgt_var = (tgt - tgt_mean).pow(2).mean(dim=1, keepdim=True)
            tgt_ln = (tgt - tgt_mean) / t.sqrt(tgt_var + self.weight_normalize_eps)
            diff = tgt_ln - src_ln
            # also do a centering on the diff
            # this time might want to do batch norm because the magnitude of diff signals the importance?
            # BUT if we had done LN for the activations, this information is already lost
            # So we stick with a layer norm on the diff as well
            diff_mean = diff.mean(dim=1, keepdim=True)
            diff_var = (diff - diff_mean).pow(2).mean(dim=1, keepdim=True)
            diff = (diff - diff_mean) / t.sqrt(diff_var + self.weight_normalize_eps)
            scale_cache = {
                "diff_mean": diff_mean,
                "diff_var": diff_var,
                "src_mean": src_mean,
                "src_var": src_var,
                "tgt_mean": tgt_mean,
                "tgt_var": tgt_var,
            }
        elif self.standardize_method == "diff_batch_norm":
            diff = tgt - src
            mu = diff.mean(0)
            diff_centered_batch = diff - mu # such that upon centering, summing over all entries are 0, and this is done via subtracting one global vector of shape (d_model,)
            norm_scale = diff_centered_batch.norm(dim=1).mean() # this is to ensure that the norm the centered diff is 1 across the dataset, instead of individually.
            # Note that this is still different from the batch norm. Here normscale is just a scalar that make sure that vectors in this batch has average norm 1.
            diff = diff_centered_batch / (norm_scale + self.weight_normalize_eps)
            scale_cache = {
                "mu": mu,
                "norm_scale": norm_scale,
            }
        else:
            raise NotImplementedError(f"Invalid standardization method: {self.standardize_method}")
        
        activations_BF = self.get_latent_activations(diff)
        sparse_activations_BF = self.apply_batchtopk(activations_BF)
        recon_BF = self.decode(sparse_activations_BF)
        
        # out = self.forward(x)
        # reconstruction = out["recon"]
        # features = out["sparse_activations"]
        # error = target - reconstruction

        self.latent_tracker.update(sparse_activations_BF)
        dead_latents = self.latent_tracker.get_dead_latents()
        error = diff - recon_BF

        loss_dict = {
            "L_reconstruction": F.mse_loss(recon_BF, diff),
            "L_aux": self.auxiliary_loss(dead_latents, error),
        }
        loss = loss_dict["L_reconstruction"] + self.aux_coeff * loss_dict["L_aux"]
        
        return loss_dict, loss, sparse_activations_BF, diff, recon_BF, scale_cache


class LatentTracker:
    def __init__(self, n_latents, dead_threshold=10_000_000, device="cuda"):
        self.n_latents = n_latents
        self.dead_threshold = dead_threshold
        self.last_activation = t.zeros(n_latents, device=device)
        self.current_step = 0
        self.device = device
        self.update_buffer = t.zeros(n_latents, device=device)

    def update(self, features):
        """Update activation tracking for each latent"""
        with t.no_grad():
            # Use where instead of boolean indexing
            self.update_buffer.zero_()
            self.update_buffer.masked_fill_(
                (features > 0).any(dim=0), self.current_step
            )
            # Use max to only update when we see a newer activation
            self.last_activation = t.maximum(
                self.last_activation, self.update_buffer
            )
            self.current_step += features.shape[0]

    def get_dead_latents(self):
        """Return boolean mask of dead latents"""
        with t.no_grad():
            steps_since_activation = self.current_step - self.last_activation
            return steps_since_activation >= self.dead_threshold


if __name__ == "__main__":
    # testing out different functionalities 
    # cfg = GatedSAEConfig()
    # gated_sae = GatedDiffSAE(cfg, device=t.device("cuda"))
    # x = t.randn(10, 768*2).to(gated_sae.device)
    # loss_dict, loss, acts_post, diff, diff_reconstructed = gated_sae(x)
    # print(loss_dict)
    # print(loss)
    # print(acts_post.shape)
    # print(diff_reconstructed.shape)

    # gated_sae.resampling(x, t.zeros(1024, 1024), 0.5)
    
    cfg = BatchTopKSAEConfig(
        d_model=768,
        d_sae=1024,
        k=10,
        device=t.device("cuda"),
        standardize_method="diff_batch_norm",
        weight_normalize_eps=1e-6,
        aux_coeff=0.01,
    )
    batch_topk_sae = BatchTopKSAE(cfg)
    x = t.randn(10, 768*2).to(cfg.device)
    loss_dict, loss, sparse_activations_BF, diff, recon_BF, scale_cache = batch_topk_sae(x)
    print(loss_dict)
    print(loss)
    print(sparse_activations_BF.shape)
    print(diff.shape)
    print(recon_BF.shape)


