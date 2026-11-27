"""Write the lerobot-policy-server config for one of our checkpoints (remote inference).

lerobot-policy-server needs every input spelled out: the name the model expects, the shape and
dtype the robot sends, and the action's joint order. All of that is already in the checkpoint
(rename map, input features) and in the dataset's info.json (camera size, joint names), so this
script reads it from there instead of writing it by hand.

    python scripts/make_server_config.py outputs/train/pi05_so101_pick_place/pretrained_model \
        --model-path /workspace/pi05/pretrained_model --out outputs/server/pi05.json

--model-path is where the checkpoint will sit on the machine that runs the server (defaults to the
local path, for a test on this laptop). The laptop side passes the same --inference.semantics.
"""

import argparse
import json
from pathlib import Path

SEMANTICS = "so101_follower_v1"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="local pretrained_model folder")
    ap.add_argument("--dataset", default="data/Yufeng1216/so101_pick_place", help="training dataset (for shapes and joint names)")
    ap.add_argument("--model-path", help="checkpoint path on the server machine (default: the local path)")
    ap.add_argument("--deployment", help="deployment name (default: policy type, e.g. pi05)")
    ap.add_argument("--port", type=int, default=7447)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ckpt = Path(a.checkpoint)
    policy = json.loads((ckpt / "config.json").read_text(encoding="utf-8"))
    steps = json.loads((ckpt / "policy_preprocessor.json").read_text(encoding="utf-8"))["steps"]
    rename = next((s["config"]["rename_map"] for s in steps if s["registry_name"] == "rename_observations_processor"), {})
    meta = json.loads((Path(a.dataset) / "meta" / "info.json").read_text(encoding="utf-8"))
    info = meta["features"]

    # The server drops the rename step (rename_map={}), so each input is declared under the name the
    # model expects; the laptop maps its camera names onto these with --rename_map.
    features = []
    for key, feat in info.items():
        if not key.startswith("observation."):
            continue
        name = rename.get(key, key)
        if name not in policy["input_features"]:
            raise SystemExit(f"{key} -> {name} is not an input of this checkpoint")
        if feat["dtype"] in ("video", "image"):
            features.append({"name": name, "shape": feat["shape"], "dtype": "uint8", "kind": "rgb", "semantics": SEMANTICS})
        else:
            features.append({"name": name, "shape": feat["shape"], "dtype": feat["dtype"], "kind": "tensor",
                             "names": feat["names"], "semantics": SEMANTICS})
    act = info["action"]
    cfg = {
        "deployment": a.deployment or policy["type"],
        "model": {"repo_or_path": a.model_path or str(ckpt.resolve()), "device": a.device},
        "execution": {"action_fps": meta["fps"]},
        "zenoh": {"listen_endpoints": [f"tcp/127.0.0.1:{a.port}"]},
        "semantics": SEMANTICS,
        "robot_type": "so101_follower",
        "features": features,
        "action_feature": {"name": "action", "shape": act["shape"], "dtype": act["dtype"], "kind": "tensor",
                           "names": act["names"], "semantics": SEMANTICS},
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"wrote {out}: deployment={cfg['deployment']}, inputs={[f['name'] for f in features]}")


if __name__ == "__main__":
    main()
