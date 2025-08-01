# DR-MDP

## Env

```sh
uv venv --python 3.11
# will run uv sync
make pip-sync
```

## Run Tests

```sh
make test-coverage
```

## Bumpversion

```sh
uv run bumpver update --patch
```

## Gym-type Environments

This section describes specifics of setting up some gym-type environments that require extra configuration.

### Box2d setup


**Missing `string.h` file**

```
# find path
mdfind -name string.h 
# expose path
export CPATH=FOUND_PATH
# e.g. /Library/Developer/CommandLineTools/SDKs/zyx.sdk/usr/include/c++/v1/
```

**Swig**

Gymanisum Box-2d environments require `swig`.

```
sudo apt install swig
```
