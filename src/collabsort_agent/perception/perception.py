"""
Perception-related definitions.
"""

from dataclasses import dataclass

import numpy as np

# from ty_extensions._internal import Unknown


@dataclass
class PerceptionConfig:
    """Perception configuration."""

    # Number of perceived columns in an observation
    n_perceived_cols: int = 3


class Perceiver:
    """Class implementing the agent perception sense"""

    def __init__(
        self,
        config: PerceptionConfig,
        treadmill_rows: list[int],
    ) -> None:
        self.config = config
        self.treadmill_rows = treadmill_rows

    def get_sensory_state(self, obs: dict) -> np.ndarray:
        """Flatten an observation into a vector: the sensory state"""

        state_features: list[int] = []

        # Agent features
        agent: dict = obs["self"]
        state_features.extend(self._get_features(agent))
        agent_col: int = agent["coords"][1]

        # Robot features
        robot: dict = obs["robot"]
        state_features.extend(self._get_features(robot))

        # Build a dict keyed by (row, col) for O(1) object lookup
        objects: tuple[dict] = obs["moving_objects"]
        obj_map: dict = {(obj["coords"][0], obj["coords"][1]): obj for obj in objects}

        perceived_cols = [
            agent_col + col for col in range(self.config.n_perceived_cols)
        ]
        for row in self.treadmill_rows:
            for col in perceived_cols:
                obj_found = obj_map.get((row, col))
                if obj_found:
                    state_features.extend(
                        [
                            1,  # Object present
                            obj_found["color"],
                            obj_found["shape"],
                        ]
                    )
                else:
                    state_features.extend([0, 0, 0])

        # Return a 1D array containing all features
        return np.array(state_features, dtype=np.int32)

    def _get_features(self, arm: dict) -> list[int]:
        """Extract features from observation data for an arm (agent or robot)"""

        row: int = arm["coords"][0]
        col: int = arm["coords"][1]
        picked_object: int = arm["picked_object"]
        collision_penalty: int = arm["collision_penalty"]

        return [row, col, picked_object, collision_penalty]
