#!/bin/bash
set -xe

DIR=$(dirname $0)
PARENT_DIR=$DIR/../..
BASE=drmdp

TIMESTAMP=`date +%s`
python $PARENT_DIR/src/$BASE/workflows/controljob.py \
    --num-runs=3 \
    --num-episodes=10 \
    --output-dir=$HOME/fs/$BASE/workflows/controljob/logs/$TIMESTAMP \
    --task-prefix $TIMESTAMP \
    --log-episode-frequency=1
