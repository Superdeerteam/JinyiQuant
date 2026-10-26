"""Write a fixed-weight score-blend checkpoint from frozen single-model checkpoints.

Members keep their own fitted ``target_scale`` and missing-history fallback, so the
blend only fixes the member weights; no refitting, rank transform or normalization is
introduced. The output is the schema-4 checkpoint that ``TLOBAdapter.load`` and the
platform package consume. Members may use either torch trunk (``modified_tlob`` or
``mlplob``), including a mixture of both, and may differ in lookback; they must share
the rest of preprocessing, prediction scale and the common recipe keys, while
trunk-specific keys must match within each trunk group. The blend is built once at the
longest member lookback and each member scores its trailing rows.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from jinyiquant.io import file_sha256, utc_now  # noqa: E402
from modified_tlob.adapter import (  # noqa: E402
    BACKBONE_SHARED_KEYS,
    BACKBONES,
    BLEND_KIND,
    BLEND_SCHEMA_VERSION,
    BLEND_SHARED_KEYS,
)

ALIASES = "name=path:weight"


def parse_member(text):
    name, _, rest = text.partition("=")
    path, _, weight = rest.rpartition(":")
    if not name or not path or not weight:
        raise ValueError(f"Member must be {ALIASES}: {text}")
    return name, Path(path), float(weight)


def parse_expected(text):
    name, _, digest = text.partition("=")
    if not name or not digest:
        raise ValueError(f"Expected sha must be name=sha256: {text}")
    return name, digest


def load_payload(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("kind") not in BACKBONES or payload.get("schema_version") not in (1, 2, 3):
        raise ValueError(f"{path}: expected a plain torch backbone checkpoint")
    return payload


def shared_preprocessing(payload):
    """Payload preprocessing minus the lookback, which the blend slices per member."""
    return {key: value for key, value in payload["preprocessing"].items() if key != "lookback"}


def validate(members):
    reference = members[0][2]
    kinds = [payload["kind"] for _, _, payload in members]
    if any(kind not in BACKBONES for kind in kinds):
        raise ValueError("Blend members must be plain torch backbone checkpoints")
    for name, _, payload in members:
        for key in BLEND_SHARED_KEYS:
            left = payload["model_config"].get(key)
            right = reference["model_config"].get(key)
            if left != right:
                raise ValueError(f"{name}: model {key}={left!r} differs from {members[0][0]} ({right!r})")
        if shared_preprocessing(payload) != shared_preprocessing(reference):
            raise ValueError(f"{name}: preprocessing differs from {members[0][0]}")
        lookback = payload["preprocessing"].get("lookback")
        if type(lookback) is not int or lookback < 4:
            raise ValueError(f"{name}: lookback must be an integer of at least 4")
        if payload.get("target_scale") != reference.get("target_scale"):
            raise ValueError(f"{name}: target_scale differs from {members[0][0]}")
    groups = {}
    for name, path, payload in members:
        groups.setdefault(payload["kind"], []).append((name, path, payload))
    for kind, group in groups.items():
        for key in BACKBONE_SHARED_KEYS[kind]:
            for name, _, payload in group[1:]:
                left = payload["model_config"].get(key)
                right = group[0][2]["model_config"].get(key)
                if left != right:
                    raise ValueError(f"{name}: model {key}={left!r} differs from {group[0][0]} ({right!r})")


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build(members, expected, *, experiment_id, note=""):
    names = [name for name, _, _ in members]
    if len(members) < 2 or len(set(names)) != len(names):
        raise ValueError("At least two uniquely named members are required")
    weights = [weight for _, _, weight in members]
    if any(not 0 <= weight <= 1 for weight in weights) or abs(sum(weights) - 1) > 1e-9:
        raise ValueError("Member weights must be in [0, 1] and sum to one")
    entries = []
    for name, path, weight in members:
        payload = load_payload(path)
        digest = file_sha256(path)
        if name in expected and digest != expected[name]:
            raise ValueError(f"{name}: weight checksum changed: {digest}")
        entries.append({"name": name, "weight": weight, "sha256": digest, "payload": payload})
    validate([(name, path, entry["payload"]) for (name, path, _), entry in zip(members, entries)])
    kinds = [entry["payload"]["kind"] for entry in entries]
    return {
        "schema_version": BLEND_SCHEMA_VERSION, "kind": BLEND_KIND,
        "members": entries,
        "blend": {"method": "score_weighted_mean", "weights": dict(zip(names, weights)),
                  "units": "member_target_scale_restored_return_score",
                  "fallback": "weighted sum of each member's training-mean fallback",
                  "kind": kinds[0] if len(set(kinds)) == 1 else "mixed",
                  "member_kinds": dict(zip(names, kinds))},
        "created_at_utc": utc_now(), "experiment_id": experiment_id,
        "code": {"commit": git_commit()},
        "source": {"builder": "scripts/build_blend_weights.py", "note": note},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member", action="append", required=True, metavar=ALIASES,
                        help="name=weights.json:weight, repeatable; weights must sum to one")
    parser.add_argument("--expected-sha256", action="append", default=[], metavar="NAME=SHA256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--note", default="")
    args = parser.parse_args(argv)
    members = [parse_member(text) for text in args.member]
    expected = dict(parse_expected(text) for text in args.expected_sha256)
    payload = build(members, expected, experiment_id=args.experiment_id, note=args.note)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False,
                                      separators=(",", ":")), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sha256": file_sha256(args.output),
                      "bytes": args.output.stat().st_size,
                      "members": {entry["name"]: entry["weight"] for entry in payload["members"]},
                      "member_sha256": {entry["name"]: entry["sha256"] for entry in payload["members"]}},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
