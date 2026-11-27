"""Start lerobot-policy-server for pi0.5 (runs inside the HF Job; see serve_pi05.py).

LeRobot 0.7's policy server checks that the declared inputs match the checkpoint's input_features
exactly (names and shapes). pi0.5 lists three cameras at 3x224x224 and a 32-dim state, but at run
time it already resizes images with padding, pads the state to 32 and masks missing cameras, exactly
as in local inference (SmolVLA overrides the check the same way; pi0.5 does not yet). So this relaxes
only the pre-serving check, the same rules as SmolVLA's override; the model and processors are unchanged.
"""

import sys

from lerobot.configs import FeatureType
from lerobot.policies.pi05.modeling_pi05 import PI05Policy


def validate_chunk_input_features(self, features) -> None:
    expected = self.config.input_features or {}
    supplied = {f.name: f for f in features}
    required = {n for n, f in expected.items() if f.type != FeatureType.VISUAL}
    if not required <= supplied.keys() or not supplied.keys() <= expected.keys():
        raise ValueError("pi0.5 needs every non-camera input and only known camera names")
    cams = [n for n in supplied if expected[n].type == FeatureType.VISUAL]
    if not cams:
        raise ValueError("pi0.5 needs at least one camera")
    for n in cams:
        f = supplied[n]
        if f.kind != "rgb" or len(f.shape) != 3 or f.shape[-1] != 3:
            raise ValueError(f"camera {n} must be an HxWx3 RGB image")
    state = supplied.get("observation.state")
    if state is not None and (len(state.shape) != 1 or state.shape[0] > self.config.max_state_dim):
        raise ValueError("state must be a vector no longer than max_state_dim")


PI05Policy.validate_chunk_input_features = validate_chunk_input_features

if __name__ == "__main__":
    from lerobot.scripts.lerobot_policy_server import main

    sys.argv = ["lerobot-policy-server", *sys.argv[1:]]
    main()
