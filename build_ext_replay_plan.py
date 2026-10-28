import glob, json, os

base = json.load(open("configs/featnorm_replay_20261028_plan.json"))
plan = dict(base)
plan["experiment_id"] = "featnorm-ext-replay-20261028"
plan["hypothesis"] = ("The six 132d extension-seed runs (seeds 53/59/61 x two arms) replayed on the same "
                      "five declared development months, reporting the declared replication of the registered "
                      "11/29/47 comparison. The registered decision stands on 11/29/47.")
plan["output_root"] = "/root/runs/featnorm-ext-replay-20261028"
versions, pairs = [], []
for seed in (53, 59, 61):
    for arm, tag in (("control", "c"), ("pricerel", "p")):
        run = sorted(glob.glob("/root/runs/featnorm-ext-%s-seed%d/*/" % (arm, seed)))[-1]
        for point, short in (("1.32", "132"), ("2.64", "264")):
            path = os.path.join(run, "modified_tlob_weights-data-%s.json" % point)
            if not os.path.exists(path):
                raise SystemExit("missing %s" % path)
            versions.append({"name": "%s-s%d-%s" % (tag, seed, short),
                             "label": "%s seed%d data-%s" % (arm, seed, point),
                             "weights": path})
    for short in ("132", "264"):
        pairs.append(["p-s%d-%s" % (seed, short), "c-s%d-%s" % (seed, short)])
plan["versions"] = versions
plan["pairs"] = pairs
json.dump(plan, open("configs/featnorm_ext_replay_plan.json", "w"), indent=1, ensure_ascii=False)
print("versions", len(versions), "pairs", len(pairs))
