import logging

import torch
from torch import nn

import boltz.model.layers.initialize as init
from boltz.model.layers.pairformer import PairformerNoSeqModule
from boltz.model.modules.encodersv2 import PairwiseConditioning
from boltz.model.modules.transformersv2 import DiffusionTransformer
from boltz.model.modules.utils import LinearNoBias

logger = logging.getLogger(__name__)


class GaussianSmearing(torch.nn.Module):
    """Gaussian smearing."""

    def __init__(
        self,
        start: float = 0.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
    ) -> None:
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.num_gaussians = num_gaussians
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        shape = dist.shape
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2)).reshape(
            *shape, self.num_gaussians
        )


class AffinityModule(nn.Module):
    """Algorithm 31"""

    def __init__(
        self,
        token_s,
        token_z,
        pairformer_args: dict,
        transformer_args: dict,
        num_dist_bins=64,
        max_dist=22,
        use_cross_transformer: bool = False,
        groups: dict = {},
    ):
        super().__init__()
        boundaries = torch.linspace(2, max_dist, num_dist_bins - 1)
        self.register_buffer("boundaries", boundaries)
        self.dist_bin_pairwise_embed = nn.Embedding(num_dist_bins, token_z)
        init.gating_init_(self.dist_bin_pairwise_embed.weight)

        self.s_to_z_prod_in1 = LinearNoBias(token_s, token_z)
        self.s_to_z_prod_in2 = LinearNoBias(token_s, token_z)

        self.z_norm = nn.LayerNorm(token_z)
        self.z_linear = LinearNoBias(token_z, token_z)

        self.pairwise_conditioner = PairwiseConditioning(
            token_z=token_z,
            dim_token_rel_pos_feats=token_z,
            num_transitions=2,
        )

        self.pairformer_stack = PairformerNoSeqModule(token_z, **pairformer_args)
        self.affinity_heads = AffinityHeadsTransformer(
            token_z,
            transformer_args["token_s"],
            transformer_args["num_blocks"],
            transformer_args["num_heads"],
            transformer_args["activation_checkpointing"],
            False,
            groups=groups,
        )

    def forward(
        self,
        s_inputs,
        z,
        x_pred,
        feats,
        multiplicity=1,
        use_kernels=False,
        distogram_mask_mode="none",
        distance_cutoff=None,
    ):
        z = self.z_linear(self.z_norm(z))
        z = z.repeat_interleave(multiplicity, 0)

        z = (
            z
            + self.s_to_z_prod_in1(s_inputs)[:, :, None, :]
            + self.s_to_z_prod_in2(s_inputs)[:, None, :, :]
        )

        token_to_rep_atom = feats["token_to_rep_atom"]
        token_to_rep_atom = token_to_rep_atom.repeat_interleave(multiplicity, 0)
        if len(x_pred.shape) == 4:
            B, mult, N, _ = x_pred.shape
            x_pred = x_pred.reshape(B * mult, N, -1)
        else:
            BM, N, _ = x_pred.shape
            B = BM // multiplicity
            mult = multiplicity
        x_pred_repr = torch.bmm(token_to_rep_atom.float(), x_pred)
        d = torch.cdist(x_pred_repr, x_pred_repr)

        distogram = (d.unsqueeze(-1) > self.boundaries).sum(dim=-1).long()
        distogram = self.dist_bin_pairwise_embed(distogram)

        # ── Debug: distogram BEFORE masking ───────────────────────
        if distogram_mask_mode != "none":
            _norm_pre = distogram.norm().item()
            _nonzero_pre = (distogram.abs() > 1e-8).sum().item()
            _numel = distogram.numel()
            logger.info(
                f"[distogram-debug] mode={distogram_mask_mode} "
                f"BEFORE masking: shape={list(distogram.shape)} "
                f"norm={_norm_pre:.4f} "
                f"nonzero={_nonzero_pre}/{_numel} "
                f"mean={distogram.mean().item():.6f} "
                f"std={distogram.std().item():.6f}"
            )

        # ── Distogram masking for ablation studies ────────────────
        if distogram_mask_mode != "none":
            pad_mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
            rec = (feats["mol_type"] == 0).repeat_interleave(multiplicity, 0) * pad_mask
            lig = (
                feats["affinity_token_mask"]
                .repeat_interleave(multiplicity, 0)
                .to(torch.bool)
            ) * pad_mask

            if distogram_mask_mode == "zero_all":
                # Zero entire embedded distogram
                distogram = torch.zeros_like(distogram)
            elif distogram_mask_mode == "zero_cross":
                # Zero protein-ligand and ligand-protein cross pairs
                cross = (
                    lig[:, :, None] * rec[:, None, :]
                    + rec[:, :, None] * lig[:, None, :]
                )
                distogram = distogram * (1 - cross.unsqueeze(-1).float())
            elif distogram_mask_mode == "zero_ligand":
                # Zero all pairs involving the ligand
                any_lig = lig[:, :, None] | lig[:, None, :]
                distogram = distogram * (~any_lig).unsqueeze(-1).float()
            elif distogram_mask_mode == "zero_receptor":
                # Zero receptor-receptor pairs only
                rec_rec = rec[:, :, None] * rec[:, None, :]
                distogram = distogram * (1 - rec_rec.unsqueeze(-1).float())
            elif distogram_mask_mode == "distance_cutoff":
                # Zero distogram entries for pairs beyond the cutoff
                cutoff = distance_cutoff if distance_cutoff is not None else 8.0
                beyond = (d > cutoff).unsqueeze(-1).float()
                distogram = distogram * (1 - beyond)

            # ── Debug: distogram AFTER masking ────────────────────
            _norm_post = distogram.norm().item()
            _nonzero_post = (distogram.abs() > 1e-8).sum().item()
            _frac_zeroed = 1.0 - (_nonzero_post / max(_nonzero_pre, 1))
            logger.info(
                f"[distogram-debug] mode={distogram_mask_mode} "
                f"AFTER masking: "
                f"norm={_norm_post:.4f} "
                f"nonzero={_nonzero_post}/{_numel} "
                f"fraction_zeroed={_frac_zeroed:.4f} "
                f"mean={distogram.mean().item():.6f} "
                f"std={distogram.std().item():.6f}"
            )
            # Mask region breakdown
            _n_rec = rec.sum().item()
            _n_lig = lig.sum().item()
            logger.info(
                f"[distogram-debug] token counts: "
                f"receptor={_n_rec:.0f} ligand={_n_lig:.0f} "
                f"total_tokens={pad_mask.sum().item():.0f}"
            )

        # ── Debug: confirm this is what enters pairwise_conditioner ──
        logger.debug(
            f"[distogram-debug] feeding to pairwise_conditioner: "
            f"norm={distogram.norm().item():.4f} "
            f"shape={list(distogram.shape)}"
        )

        z = z + self.pairwise_conditioner(z_trunk=z, token_rel_pos_feats=distogram)

        pad_token_mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
        rec_mask = (feats["mol_type"] == 0).repeat_interleave(multiplicity, 0)
        rec_mask = rec_mask * pad_token_mask
        lig_mask = (
            feats["affinity_token_mask"]
            .repeat_interleave(multiplicity, 0)
            .to(torch.bool)
        )
        lig_mask = lig_mask * pad_token_mask
        cross_pair_mask = (
            lig_mask[:, :, None] * rec_mask[:, None, :]
            + rec_mask[:, :, None] * lig_mask[:, None, :]
            + lig_mask[:, :, None] * lig_mask[:, None, :]
        )
        z = self.pairformer_stack(
            z,
            pair_mask=cross_pair_mask,
            use_kernels=use_kernels,
        )

        out_dict = {}

        # affinity heads
        out_dict.update(
            self.affinity_heads(z=z, feats=feats, multiplicity=multiplicity)
        )

        return out_dict


class AffinityHeadsTransformer(nn.Module):
    def __init__(
        self,
        token_z,
        input_token_s,
        num_blocks,
        num_heads,
        activation_checkpointing,
        use_cross_transformer,
        groups={},
    ):
        super().__init__()
        self.affinity_out_mlp = nn.Sequential(
            nn.Linear(token_z, token_z),
            nn.ReLU(),
            nn.Linear(token_z, input_token_s),
            nn.ReLU(),
        )

        self.to_affinity_pred_value = nn.Sequential(
            nn.Linear(input_token_s, input_token_s),
            nn.ReLU(),
            nn.Linear(input_token_s, input_token_s),
            nn.ReLU(),
            nn.Linear(input_token_s, 1),
        )

        self.to_affinity_pred_score = nn.Sequential(
            nn.Linear(input_token_s, input_token_s),
            nn.ReLU(),
            nn.Linear(input_token_s, input_token_s),
            nn.ReLU(),
            nn.Linear(input_token_s, 1),
        )
        self.to_affinity_logits_binary = nn.Linear(1, 1)

    def forward(
        self,
        z,
        feats,
        multiplicity=1,
    ):
        pad_token_mask = (
            feats["token_pad_mask"].repeat_interleave(multiplicity, 0).unsqueeze(-1)
        )
        rec_mask = (
            (feats["mol_type"] == 0).repeat_interleave(multiplicity, 0).unsqueeze(-1)
        )
        rec_mask = rec_mask * pad_token_mask
        lig_mask = (
            feats["affinity_token_mask"]
            .repeat_interleave(multiplicity, 0)
            .to(torch.bool)
            .unsqueeze(-1)
        ) * pad_token_mask
        cross_pair_mask = (
            lig_mask[:, :, None] * rec_mask[:, None, :]
            + rec_mask[:, :, None] * lig_mask[:, None, :]
            + (lig_mask[:, :, None] * lig_mask[:, None, :])
        ) * (
            1
            - torch.eye(lig_mask.shape[1], device=lig_mask.device)
            .unsqueeze(-1)
            .unsqueeze(0)
        )

        g = torch.sum(z * cross_pair_mask, dim=(1, 2)) / (
            torch.sum(cross_pair_mask, dim=(1, 2)) + 1e-7
        )

        g = self.affinity_out_mlp(g)

        affinity_pred_value = self.to_affinity_pred_value(g).reshape(-1, 1)
        affinity_pred_score = self.to_affinity_pred_score(g).reshape(-1, 1)
        affinity_logits_binary = self.to_affinity_logits_binary(
            affinity_pred_score
        ).reshape(-1, 1)
        out_dict = {
            "affinity_pred_value": affinity_pred_value,
            "affinity_logits_binary": affinity_logits_binary,
        }
        return out_dict
