"""Parameter efficiency accounting for Population-DTI-INR (no architecture change)."""
from __future__ import annotations

from typing import Any

import torch.nn as nn

from models.population_dti_inr import PopulationDTIINR


def count_theta_parameters(model: PopulationDTIINR) -> int:
    return int(sum(p.numel() for p in model.theta_parameters()))


def count_train_latent_parameters(model: PopulationDTIINR) -> int:
    if model.latents is None:
        return 0
    return int(sum(p.numel() for p in model.latents.parameters()))


def count_adaptable_parameters(model: PopulationDTIINR) -> int:
    """Per unseen subject: only z_new (latent_dim)."""
    return int(model.latent_dim)


def parameter_efficiency_report(model: PopulationDTIINR) -> dict[str, Any]:
    n_theta = count_theta_parameters(model)
    n_latents = count_train_latent_parameters(model)
    n_adapt = count_adaptable_parameters(model)
    ratio = float(n_adapt) / float(n_theta) if n_theta > 0 else float("nan")
    return {
        "total_theta_parameters": n_theta,
        "subject_latent_parameters_train_table": n_latents,
        "adaptable_parameter_count": n_adapt,
        "adaptation_parameter_ratio": ratio,
        "latent_dim": int(model.latent_dim),
        "n_train_subjects_in_table": len(model.latents.subject_ids) if model.latents else 0,
        "formula": "adaptation_parameter_ratio = adaptable_parameter_count / total_theta_parameter_count",
    }


def count_module_parameters(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))
