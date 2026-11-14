#!/bin/bash
set -xe

DIR=$(dirname $0)
PARENT_DIR=$DIR/../..
BASE=drmdp

TIMESTAMP=`date +%s`
ray job submit \
    --address http://127.0.0.1:8265 \
    --working-dir $PARENT_DIR \
    --runtime-env-json='{"py_modules":["src/drmdp"], "excludes": [".git"]}' \
    -- \
    python $PARENT_DIR/src/$BASE/workflows/control/job.py \
        --num-runs=2 \
        --num-episodes=100 \
        --output-dir=$HOME/fs/$BASE/control/fnapprox/gaussian/$TIMESTAMP \
        --task-prefix $TIMESTAMP \
        --log-episode-frequency=5 \
        --problem-set gaussian-illustration \
        --grids-file $PARENT_DIR/assets/grids.json \
        --use-seed
