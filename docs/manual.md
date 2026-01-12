## Discrete state and actions

It is better to use the environment's defined state and action mappings, to get precise estimates.
In some `gymnasium` environments `P` variable encode more states than actually exist.
With `rlplg`, the `transition` functions maps states to a complete set, avoid such cases.

For that, load the environment with a `FlatGridCoordObsWrapper`.
