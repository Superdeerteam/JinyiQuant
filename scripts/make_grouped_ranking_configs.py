"""Freeze the grouped-sampling MSE control and MSE+pairwise-ranking arms from a base recipe.

Usage: python scripts/make_grouped_ranking_configs.py <base.json> <tag> <seed,seed,...> [weight]
The base must be the plain shuffle/MSE recipe selected for promotion; the two arms differ
only in the objective, so a same-seed paired comparison isolates the ranking term.
"""

import json
import sys
from pathlib import Path

BASE = Path(sys.argv[1])
TAG = sys.argv[2]
SEEDS = [int(value) for value in sys.argv[3].split(",")]
WEIGHT = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
if WEIGHT <= 0:
    raise SystemExit("ranking weight must be positive for the ranking arm")

base = json.loads(BASE.read_text(encoding="utf-8"))
model = base["model"]
if model.get("sampling", "shuffle") != "shuffle" or model.get("ranking_loss", "none") != "none":
    raise SystemExit("base recipe must be plain shuffle sampling with no ranking term")
BATCH = int(model["batch_size"])
written = []
for seed in SEEDS:
    for arm, ranking in (("mse", False), ("rank", True)):
        config = json.loads(json.dumps(base))
        config["experiment_id"] = f"{TAG}-group{BATCH}-{arm}-seed{seed}"
        config["output_root"] = f"../runs/{config['experiment_id']}"
        config["seed"] = seed
        config["hypothesis"] = (
            f"Same recipe as {base['experiment_id']} with identical budgets, data, labels and seed {seed}; only the "
            f"sampler changes to one same-timestamp cross-section of {BATCH} rows per step"
            + (f", and the objective adds {WEIGHT:g} x mean pairwise-hinge ranking over same-timestamp pairs "
               "(paper Eq. 8 form, weight fixed before training)" if ranking else
               " with pure MSE as the shared-sampler control")
            + ". Paired same-seed comparison isolates the ranking term; single validation window, local approximation.")
        model = config["model"]
        model["sampling"] = "timestamp_grouped"
        model["ranking_loss"] = "pairwise_hinge" if ranking else "none"
        model["ranking_loss_weight"] = WEIGHT if ranking else 0.0
        path = BASE.parent / f"{config['experiment_id']}.json"
        path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append(str(path))
print("\n".join(written))
