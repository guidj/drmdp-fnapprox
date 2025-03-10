SCALE = "scale"
GAUSSIAN_MIX = "gaussian-mix"
TILES = "tiles"
RANDOM = "random"
DEFAULT_PARAMS_GRID = {
    "n_components": range(4, 12),
    "covariance_type": ["spherical", "tied", "diag", "full"],
}
DEFAULT_GM_STEPS = 100_000
DEFAULT_TILING_DIM = 6
DEFAULT_HASH_DIM = 512
