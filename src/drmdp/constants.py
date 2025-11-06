"""
Constants and default values.
"""

IDENTITY = "identity"
SCALE = "scale"
GAUSSIAN_MIX = "gaussian-mix"
TILES = "tiles"
SPLICED_TILES = "spliced-tiles"
RANDOM = "random"
CLUSTER_CENTROID = "cluster-c"
FLAT_GRID_COORD = "flat-grid-coord"
DEFAULT_PARAMS_GRID = {
    "n_components": range(4, 12),
    "covariance_type": ["spherical", "tied", "diag", "full"],
}
DEFAULT_GM_STEPS = 10_000
DEFAULT_TILING_DIM = 6
DEFAULT_HASH_DIM = 512
DEFAULT_CLUSTER_STEPS = 10_000


OPTIONS_POLICY = "options"
SINGLE_STEP_POLICY = "single-step"

DEFAULT_BATCH_SIZE = 1

SARSA = "sarsa"
