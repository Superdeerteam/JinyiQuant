"""Freeze the grouped-sampling MSE control and MSE+pairwise-ranking arms from a base recipe."""

import json
import sys
from pathlib import Path

BASE = Path(sys.argv[1])
TAG = sys.argv[2]
SEEDS = [int(value) for value in sys.argv[3].split(",")]
WEIGHT = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0

base = json.loads(BASE.read_text(encoding="utf-8"))
written = []
for seed in SEEDS:
    for arm, ranking in (("mse", False), ("rank", True)):
        config = json.loads(json.dumps(base))
        config["experiment_id"] = f"{TAG}-group32-{arm}-seed{seed}"
        config["output_root"] = f"../runs/{config['experiment_id']}"
        config["seed"] = seed
        config["hypothesis"] = (
            f"Same recipe as {base['experiment_id']} with identical budgets, data, labels and seed {seed}; only the "
            "sampler changes to one same-timestamp cross-section per step"
            + (f", and the objective adds {WEIGHT:g} x mean pairwise-hinge ranking over same-timestamp pairs "
               "(Feng et al. 2018 form)" if ranking else " with pure MSE as the shared-sampler control")
            + ". Paired same-seed comparison isolates the ranking term; single validation window, local approximation.")
        config["model"]["sampling"] = "timestamp_grouped"
        config["model"]["ranking_loss"] = "pairwise_hinge" if ranking else "none"
        config["model"]["ranking_loss_weight"] = WEIGHT if ranking else 0.0
        path = BASE.parent / f"{config['experiment_id']}.json"
        path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append(str(path))
print("\n".join(written))
