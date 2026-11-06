"""
Constants and default values.
"""

FINITE = "finite"
IDENTITY = "identity"
FINITE_IDENTITY = f"{FINITE}-{IDENTITY}"
SCALE = "scale"
FINITE_SCALE = f"{FINITE}-{SCALE}"
GAUSSIAN_MIX = "gaussian-mix"
FINITE_GAUSSIAN_MIX = f"{FINITE}-{GAUSSIAN_MIX}"
TILES = "tiles"
FINITE_TILES = f"{FINITE}-{TILES}"
SPLICED_TILES = "spliced-tiles"
FINITE_SPLICED_TILES = f"{FINITE}-{SPLICED_TILES}"
RANDOM_VEC = "random"
FINITE_RANDOM_VEC = f"{FINITE}-{RANDOM_VEC}"
CLUSTER_CENTROID = "cluster-c"
FLAT_GRID_COORD = "flat-grid-coord"
FINITE_FLAT_GRID_COORD = f"{FINITE}-{FLAT_GRID_COORD}"
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

# estimation-buffer capacity as a multiple of the estimator's feature
# dimension: the buffer retains DEFAULT_ESTIMATION_BUFFER_MULT * mdim completed
# delay windows. Value shared by every experiment family that estimates rewards
# from aggregate windows, so estimation error is compared at one sample budget.
DEFAULT_ESTIMATION_BUFFER_MULT = 100

SARSA = "sarsa"


def finite(suffix: str) -> str:
    return f"finite-{suffix}"
