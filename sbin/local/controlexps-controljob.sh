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
    python $PARENT_DIR/src/$BASE/workflows/controljob.py \
        --num-runs=1 \
        --num-episodes=100 \
        --output-dir=$HOME/fs/$BASE/control/fnapprox/logs/$TIMESTAMP \
        --task-prefix $TIMESTAMP \
        --log-episode-frequency=5 \
        --problem-set grid-world \
        --grids-file $PARENT_DIR/assets/grids.json \
        --use-seed
