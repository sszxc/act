# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import os

import torch
from torch import nn
from PIL import Image
from torch.autograd import Variable
from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer

import numpy as np

import IPython
e = IPython.embed


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


# FiLM-after feature map dump for offline video stitching. Off by default (would block automated rollout).
# Enable with: ACT_FILM_VIZ_SAVE=1
_FILM_VIZ_SAVE = os.environ.get("ACT_FILM_VIZ_SAVE", "").lower() in ("1", "true", "yes")
_FILM_VIZ_OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "tmp",
    "film_features_after_film",
)
_FILM_VIZ_MODE = "mean"  # "max" or "mean"


class DETRVAE(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names, action_dim=None, latent_z_dim=32):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension (qpos / observation)
            num_queries: number of object queries
            camera_names: list of camera names
            action_dim: policy output dimension; if None, equals state_dim (backward compatible)
        """
        super().__init__()
        if action_dim is None:
            action_dim = state_dim
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)

            # Hardcoded FiLM on visual features before Transformer.
            # Edit these values directly in this file as needed.
            self.register_buffer("visual_film_gamma", torch.ones(hidden_dim))
            self.register_buffer("visual_film_beta", torch.zeros(hidden_dim))
            # self.visual_film_beta[: hidden_dim // 2] = -0.5
            # self.visual_film_gamma[: hidden_dim // 2] = 0.5
            # self.visual_film_gamma[hidden_dim // 2 :] = 0.5
            # self.visual_film_beta[hidden_dim // 2 :] += 0.01 * torch.randn(hidden_dim - hidden_dim // 2)
            self._film_viz_frame_idx = 0

            # --- PCA-bottleneck FiLM (opt-in, independent of the plain per-channel FiLM above) ---
            # Hard bottleneck: src is encoded into a k-dim PCA subspace (W, mu fit offline from
            # recorded activations by fit_film_pca.py), FiLM-modulated there (film_pca_gamma/beta,
            # k-dim each, are the only free/searched params), then decoded straight back to
            # hidden_dim and used as `src` outright — it REPLACES src, it does not compose with
            # visual_film_gamma/beta above. Disabled by default (film_pca_k == 0); see
            # load_film_pca() to install a basis, and forward()'s film_pca_gamma/film_pca_beta args
            # for per-sample batched search.
            #
            # NOTE: because the k-dim subspace only spans part of hidden_dim's variance,
            # x_hat = W @ (W^T (x - mu)) + mu != x in general, even at the identity setting
            # gamma=1, beta=0 — reconstruction loss is inherent to this hard bottleneck, not a
            # bug. A residual/soft variant that preserves the orthogonal complement (x_hat = x +
            # W @ ((gamma-1) * z + beta)) is a possible future extension, not implemented here.
            self.film_pca_k = 0
            self.register_buffer("film_pca_W", torch.zeros(hidden_dim, 0))
            self.register_buffer("film_pca_mu", torch.zeros(hidden_dim))
            self.register_buffer("film_pca_gamma", torch.zeros(0))
            self.register_buffer("film_pca_beta", torch.zeros(0))
        else:
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # --- PCA-bottleneck FiLM at other encoder/decoder-boundary insertion points ---
        # Same encode->modulate->decode mechanism as film_pca_* above (see load_film_pca()),
        # just applied to a different hidden_dim-wide tensor. Independent of film_pca_* and of
        # each other — each has its own basis/free-params and is a no-op until load_film_pca()
        # installs a basis for that target (film_pca_{target}_k == 0 by default). Registered
        # unconditionally (not just when backbones is not None) since both `memory` and `hs`
        # exist regardless of the visual branch.
        #   "memory": the Transformer encoder's output, right before the decoder reads it as
        #     `memory` — literally between "encoder" and "decoder" (see transformer.py's
        #     memory_film_fn hook). Shape (S, B, hidden_dim), S = 2 + H*W*num_cam tokens.
        #   "hs": the Transformer decoder's output, right before action_head/is_pad_head.
        #     Shape (B, num_queries, hidden_dim).
        # gamma/beta broadcast across all tokens (S or num_queries) the same way the visual
        # path's gamma/beta broadcast across H,W — see _pca_bottleneck_core().
        self.film_pca_memory_k = 0
        self.register_buffer("film_pca_memory_W", torch.zeros(hidden_dim, 0))
        self.register_buffer("film_pca_memory_mu", torch.zeros(hidden_dim))
        self.register_buffer("film_pca_memory_gamma", torch.zeros(0))
        self.register_buffer("film_pca_memory_beta", torch.zeros(0))

        self.film_pca_hs_k = 0
        self.register_buffer("film_pca_hs_W", torch.zeros(hidden_dim, 0))
        self.register_buffer("film_pca_hs_mu", torch.zeros(hidden_dim))
        self.register_buffer("film_pca_hs_gamma", torch.zeros(0))
        self.register_buffer("film_pca_hs_beta", torch.zeros(0))

        # encoder extra parameters
        self.latent_dim = latent_z_dim # final size of latent z
        self.cls_embed = nn.Embedding(1, hidden_dim) # extra cls token embedding
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim) # project action to embedding
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)  # project qpos to embedding
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim)) # [CLS], qpos, a_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(2, hidden_dim) # learned position embedding for proprio and latent

    _FILM_PCA_TARGETS = ("visual", "memory", "hs")

    def load_film_pca(self, W: torch.Tensor, mu: torch.Tensor, target: str = "visual"):
        """Install a PCA-bottleneck FiLM basis at one of three insertion points (see the
        film_pca_*/film_pca_memory_*/film_pca_hs_* buffers set up in __init__):
          - "visual" (default): visual `src` right after the CNN backbone, before the
            Transformer encoder reads it (the original mode; buffers film_pca_*).
          - "memory": the Transformer encoder's output, between encoder and decoder (buffers
            film_pca_memory_*; see transformer.py's memory_film_fn hook).
          - "hs": the Transformer decoder's output, before action_head/is_pad_head (buffers
            film_pca_hs_*).
        All three share the same encode->modulate->decode math (_pca_bottleneck_core); they
        only differ in which hidden_dim-wide tensor they're applied to.

        W: (hidden_dim, k) principal directions (columns), mu: (hidden_dim,) mean — both fit
        offline by fit_film_pca.py (--target matching this one) from recorded activations at
        the corresponding hook point. Replaces any previously loaded basis for that target;
        gamma/beta reset to identity (1, 0). Pass k=0 (W with 0 columns) to disable that
        target's PCA-bottleneck path (falls back to plain FiLM for "visual"; no-op for
        "memory"/"hs").
        """
        if target not in self._FILM_PCA_TARGETS:
            raise ValueError(f"Unknown FiLM target {target!r}; expected one of {self._FILM_PCA_TARGETS}")
        prefix = "film_pca" if target == "visual" else f"film_pca_{target}"
        hidden_dim = self.hidden_dim
        if W.ndim != 2 or W.shape[0] != hidden_dim:
            raise ValueError(f"{prefix} W shape {tuple(W.shape)} expected (hidden_dim={hidden_dim}, k)")
        if mu.shape != (hidden_dim,):
            raise ValueError(f"{prefix} mu shape {tuple(mu.shape)} expected ({hidden_dim},)")
        device = self.additional_pos_embed.weight.device
        dtype = self.additional_pos_embed.weight.dtype
        k = int(W.shape[1])
        setattr(self, f"{prefix}_k", k)
        setattr(self, f"{prefix}_W", W.detach().to(device=device, dtype=dtype).clone())
        setattr(self, f"{prefix}_mu", mu.detach().to(device=device, dtype=dtype).clone())
        setattr(self, f"{prefix}_gamma", torch.ones(k, device=device, dtype=dtype))
        setattr(self, f"{prefix}_beta", torch.zeros(k, device=device, dtype=dtype))

    @staticmethod
    def _pca_bottleneck_core(
        x_bnc: torch.Tensor,
        W: torch.Tensor,
        mu: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Shared encode->modulate->decode math for the "memory"/"hs" PCA-bottleneck FiLM
        targets (see load_film_pca()). Mirrors the inline visual-src math in forward() below,
        just generalized to a canonical (B, N, C) layout so it works for both memory's
        (S,B,C)-shaped tokens and hs's (B,Q,C)-shaped tokens without duplicating the einsum
        per shape — callers permute into (B, N, C) first (N = whatever the "token" axis is:
        sequence position, query index, etc.) and permute the result back.

        x_bnc: (B, N, C). W: (C, k). mu: (C,).
        gamma, beta: (k,) shared across the whole batch, or (B, k) per-sample (batched search,
        same convention as film_pca_gamma/beta). Broadcast across N either way (gamma/beta do
        not vary across tokens), matching the visual path's spatial broadcast over H,W.
        """
        z = torch.einsum("bnc,ck->bnk", x_bnc - mu.view(1, 1, -1), W)
        g = gamma.view(1, 1, -1) if gamma.dim() == 1 else gamma.view(gamma.shape[0], 1, -1)
        b = beta.view(1, 1, -1) if beta.dim() == 1 else beta.view(beta.shape[0], 1, -1)
        z = z * g + b
        return torch.einsum("bnk,ck->bnc", z, W) + mu.view(1, 1, -1)

    def _resolve_film_pca_override(
        self, override, default_buffer: torch.Tensor, ref: torch.Tensor
    ) -> torch.Tensor:
        """Same per-sample-override convention as the visual film_pca_gamma/beta args in
        forward(): None falls back to the installed buffer (shared across the batch); a
        provided tensor/array (k,) or (bs,k) is used as-is (cast to ref's device/dtype)."""
        if override is None:
            return default_buffer.to(dtype=ref.dtype, device=ref.device)
        t = override
        if not torch.is_tensor(t):
            t = torch.as_tensor(t, dtype=torch.float32)
        return t.to(device=ref.device, dtype=ref.dtype)

    def _apply_film_pca_memory(self, memory: torch.Tensor, gamma_override, beta_override) -> torch.Tensor:
        """Candidate-4 insertion point: modulate the Transformer encoder's output (S,B,C),
        right between encoder and decoder. No-op if no basis is loaded (film_pca_memory_k==0)."""
        if int(self.film_pca_memory_k) <= 0:
            return memory
        W = self.film_pca_memory_W.to(dtype=memory.dtype, device=memory.device)
        mu = self.film_pca_memory_mu.to(dtype=memory.dtype, device=memory.device)
        gamma = self._resolve_film_pca_override(gamma_override, self.film_pca_memory_gamma, memory)
        beta = self._resolve_film_pca_override(beta_override, self.film_pca_memory_beta, memory)
        x_bnc = memory.permute(1, 0, 2)  # (S,B,C) -> (B,S,C)
        x_hat = self._pca_bottleneck_core(x_bnc, W, mu, gamma, beta)
        return x_hat.permute(1, 0, 2)  # (B,S,C) -> (S,B,C)

    def _apply_film_pca_hs(self, hs: torch.Tensor, gamma_override, beta_override) -> torch.Tensor:
        """Candidate-5 insertion point: modulate the Transformer decoder's output (B,Q,C),
        right before action_head/is_pad_head. No-op if no basis is loaded (film_pca_hs_k==0)."""
        if int(self.film_pca_hs_k) <= 0:
            return hs
        W = self.film_pca_hs_W.to(dtype=hs.dtype, device=hs.device)
        mu = self.film_pca_hs_mu.to(dtype=hs.dtype, device=hs.device)
        gamma = self._resolve_film_pca_override(gamma_override, self.film_pca_hs_gamma, hs)
        beta = self._resolve_film_pca_override(beta_override, self.film_pca_hs_beta, hs)
        return self._pca_bottleneck_core(hs, W, mu, gamma, beta)  # already (B,Q,C)

    def forward(
        self,
        qpos,
        image,
        env_state,
        actions=None,
        is_pad=None,
        latent_z_sample=None,
        film_gamma=None,
        film_beta=None,
        film_pca_gamma=None,
        film_pca_beta=None,
        film_pca_memory_gamma=None,
        film_pca_memory_beta=None,
        film_pca_hs_gamma=None,
        film_pca_hs_beta=None,
    ):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        film_pca_gamma/film_pca_beta: optional (k,) or (bs, k) override for the visual
        PCA-bottleneck FiLM path (only used when film_pca_k > 0; see load_film_pca()). Mirrors
        film_gamma/film_beta's per-sample override convention.
        film_pca_memory_gamma/beta, film_pca_hs_gamma/beta: same override convention, for the
        "memory" (encoder-decoder boundary) and "hs" (pre-action_head) PCA-bottleneck FiLM
        targets respectively (see load_film_pca(..., target=...) and
        _apply_film_pca_memory()/_apply_film_pca_hs()). Independent of film_pca_gamma/beta.
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        ### Obtain latent z from action sequence
        if is_training:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2) # (seq+1, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device) # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0] # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            if latent_z_sample is None:
                latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            else:
                z = latent_z_sample
                if not torch.is_tensor(z):
                    z = torch.as_tensor(z, dtype=torch.float32)
                z = z.to(device=qpos.device, dtype=torch.float32)
                if z.ndim == 1:
                    if z.shape[0] != self.latent_dim:
                        raise ValueError(f"latent_z_sample dim mismatch: got {z.shape[0]}, expected {self.latent_dim}")
                    z = z.unsqueeze(0).expand(bs, -1)
                elif z.ndim == 2:
                    if z.shape[1] != self.latent_dim:
                        raise ValueError(f"latent_z_sample dim mismatch: got {z.shape[1]}, expected {self.latent_dim}")
                    if z.shape[0] == 1 and bs > 1:
                        z = z.expand(bs, -1)
                    elif z.shape[0] != bs:
                        raise ValueError(f"latent_z_sample batch mismatch: got {z.shape[0]}, expected {bs}")
                else:
                    raise ValueError(f"latent_z_sample must be 1D or 2D, got shape {tuple(z.shape)}")
                latent_sample = z
            latent_input = self.latent_out_proj(latent_sample)

        # Candidate-4 hook, built once (independent of the backbones branch below): applied
        # inside Transformer.forward() right between its encoder and decoder.
        memory_film_fn = (
            (lambda memory: self._apply_film_pca_memory(memory, film_pca_memory_gamma, film_pca_memory_beta))
            if self.film_pca_memory_k > 0 else None
        )

        if self.backbones is not None:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            for cam_id, cam_name in enumerate(self.camera_names):
                features, pos = self.backbones[0](image[:, cam_id]) # HARDCODED
                features = features[0] # take the last layer feature
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos)
            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)

            if self.film_pca_k > 0:
                # PCA-bottleneck FiLM (hard bottleneck): encode -> modulate -> decode, REPLACING
                # src outright (does not compose with visual_film_gamma/beta below). See
                # load_film_pca() / the film_pca_* buffer comments in __init__ for the math.
                if film_pca_gamma is None:
                    g_pca = self.film_pca_gamma.to(dtype=src.dtype, device=src.device)
                else:
                    g_pca = film_pca_gamma
                    if not torch.is_tensor(g_pca):
                        g_pca = torch.as_tensor(g_pca, dtype=torch.float32)
                    g_pca = g_pca.to(device=src.device, dtype=src.dtype)

                if film_pca_beta is None:
                    b_pca = self.film_pca_beta.to(dtype=src.dtype, device=src.device)
                else:
                    b_pca = film_pca_beta
                    if not torch.is_tensor(b_pca):
                        b_pca = torch.as_tensor(b_pca, dtype=torch.float32)
                    b_pca = b_pca.to(device=src.device, dtype=src.dtype)

                g_pca = g_pca.view(1, -1, 1, 1) if g_pca.ndim == 1 else g_pca.view(g_pca.shape[0], -1, 1, 1)
                b_pca = b_pca.view(1, -1, 1, 1) if b_pca.ndim == 1 else b_pca.view(b_pca.shape[0], -1, 1, 1)

                W = self.film_pca_W.to(dtype=src.dtype, device=src.device)  # (hidden_dim, k)
                mu = self.film_pca_mu.to(dtype=src.dtype, device=src.device)  # (hidden_dim,)
                z = torch.einsum("bchw,ck->bkhw", src - mu.view(1, -1, 1, 1), W)  # encode
                z = z * g_pca + b_pca  # modulate (the only free params: film_pca_gamma/beta)
                src = torch.einsum("bkhw,ck->bchw", z, W) + mu.view(1, -1, 1, 1)  # decode, replaces src
            else:
                # FiLM (feature-wise affine): src = gamma * src + beta
                # Default: use internal buffers (shared across batch).
                # Optional: allow per-sample FiLM via film_gamma/film_beta of shape (bs, hidden_dim).
                if film_gamma is None:
                    gamma = self.visual_film_gamma.view(1, -1, 1, 1).to(dtype=src.dtype, device=src.device)
                else:
                    g = film_gamma
                    if not torch.is_tensor(g):
                        g = torch.as_tensor(g, dtype=torch.float32)
                    g = g.to(device=src.device, dtype=src.dtype)
                    if g.ndim == 1:
                        g = g.view(1, -1)
                    gamma = g.view(g.shape[0], -1, 1, 1)

                if film_beta is None:
                    beta = self.visual_film_beta.view(1, -1, 1, 1).to(dtype=src.dtype, device=src.device)
                else:
                    b = film_beta
                    if not torch.is_tensor(b):
                        b = torch.as_tensor(b, dtype=torch.float32)
                    b = b.to(device=src.device, dtype=src.dtype)
                    if b.ndim == 1:
                        b = b.view(1, -1)
                    beta = b.view(b.shape[0], -1, 1, 1)

                src = src * gamma + beta

            if _FILM_VIZ_SAVE and not self.training:
                with torch.no_grad():
                    if _FILM_VIZ_MODE == "mean":
                        gray = src[0].detach().mean(dim=0).float().cpu().numpy()
                    elif _FILM_VIZ_MODE == "max":
                        gray = src[0].detach().max(dim=0)[0].float().cpu().numpy()
                    else:
                        raise ValueError(f"Invalid FiLM visualization mode: {_FILM_VIZ_MODE}")
                    g_min, g_max = float(gray.min()), float(gray.max())
                    if g_max > g_min:
                        gray_u8 = ((gray - g_min) / (g_max - g_min) * 255.0).astype(np.uint8)
                    else:
                        gray_u8 = np.zeros_like(gray, dtype=np.uint8)
                    os.makedirs(_FILM_VIZ_OUT_DIR, exist_ok=True)
                    idx = self._film_viz_frame_idx
                    self._film_viz_frame_idx = idx + 1
                    Image.fromarray(gray_u8, mode="L").save(
                        os.path.join(_FILM_VIZ_OUT_DIR, f"film_{_FILM_VIZ_MODE}_{idx:06d}.png")
                    )

            hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input,
                                   self.additional_pos_embed.weight, memory_film_fn=memory_film_fn)[0]
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1) # seq length = 2
            hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight,
                                   memory_film_fn=memory_film_fn)[0]
        if self.film_pca_hs_k > 0:
            hs = self._apply_film_pca_hs(hs, film_pca_hs_gamma, film_pca_hs_beta)
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar]


class CNNMLP(nn.Module):
    def __init__(self, backbones, state_dim, camera_names, action_dim=None):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            state_dim: robot state dimension (qpos input)
            camera_names: list of camera names
            action_dim: policy output dimension; if None, equals state_dim (backward compatible)
        """
        super().__init__()
        if action_dim is None:
            action_dim = state_dim
        self.camera_names = camera_names
        self.action_head = nn.Linear(1000, action_dim) # TODO add more
        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            backbone_down_projs = []
            for backbone in backbones:
                down_proj = nn.Sequential(
                    nn.Conv2d(backbone.num_channels, 128, kernel_size=5),
                    nn.Conv2d(128, 64, kernel_size=5),
                    nn.Conv2d(64, 32, kernel_size=5)
                )
                backbone_down_projs.append(down_proj)
            self.backbone_down_projs = nn.ModuleList(backbone_down_projs)

            mlp_in_dim = 768 * len(backbones) + state_dim
            self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=action_dim, hidden_depth=2)
        else:
            raise NotImplementedError

    def forward(self, qpos, image, env_state, actions=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        # Image observation features and position embeddings
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0] # take the last layer feature
            pos = pos[0] # not used
            all_cam_features.append(self.backbone_down_projs[cam_id](features))
        # flatten everything
        flattened_features = []
        for cam_feature in all_cam_features:
            flattened_features.append(cam_feature.reshape([bs, -1]))
        flattened_features = torch.cat(flattened_features, axis=1) # 768 each
        features = torch.cat([flattened_features, qpos], axis=1) # qpos: state_dim
        a_hat = self.mlp(features)
        return a_hat


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


def build_encoder(args):
    d_model = args.hidden_dim # 256
    dropout = args.dropout # 0.1
    nhead = args.nheads # 8
    dim_feedforward = args.dim_feedforward # 2048
    num_encoder_layers = args.enc_layers # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build(args):
    state_dim = getattr(args, 'state_dim', 14)
    action_dim = getattr(args, 'action_dim', None)

    backbones = []
    backbone = build_backbone(args)
    backbones.append(backbone)

    transformer = build_transformer(args)

    encoder = build_encoder(args)

    model = DETRVAE(
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
        action_dim=action_dim,
        latent_z_dim=args.latent_z_dim,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

def build_cnnmlp(args):
    state_dim = getattr(args, 'state_dim', 14)
    action_dim = getattr(args, 'action_dim', None)

    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)

    model = CNNMLP(
        backbones,
        state_dim=state_dim,
        camera_names=args.camera_names,
        action_dim=action_dim,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model
