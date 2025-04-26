Minimal reproduction for training a gated SAE c.f. [the Deepmind paper](https://arxiv.org/pdf/2404.16014) . Structure of the project:

```bash
sae-diff/
├─ configs/                # hydra-or-yaml configs for sweeps
│   ├─ base.yaml
│   ├─ model=gpt2.yaml
│   └─ sae/topk16k.yaml
├─ src/
│   ├─ activation_buffer.py   # fast data-loader for LM activations
│   ├─ sae_modules.py         # TopK, BatchTopK, Gated, Switch …
│   ├─ trainer.py             # generic Trainer + callbacks
│   ├─ eval.py                # CE-loss-recovered etc.
│   └─ cli.py                 # `python -m cli train --config ...`
├─ sweep.py                   # wandb / slurm launch script
└─ README.md

```


## Notes on technical details

- Original virtual batch size 256 run has FVU ~1 yet reconstruction loss was ~0.005

### Normalization methods

We have a set of vectors $\mathbf{v}, \mathbf{w}$ from base and fine-tuned model respectively. Denote $\mathbf{d}\equiv \mathbf{v}-\mathbf{w}$ as the difference vector. All of $\mathbf{v}, \mathbf{w}, \mathbf{d}$ have shape `(batch, d_model)`.

1. Divide each (batch) by a constant scalar and add a constant vector, so

$$
\mathbf{d} \to  \mathbf{d}' = \gamma (\mathbf{v} + \vec{\phi})
$$
such that the *average across the batch has zero mean and unit norm*. This means
 
 $$
\vec{\phi}=-\sum_{i=1}^{\texttt{batch}} \vec{d}_{i}, \gamma = \| \mathbf{v} + \|
 $$