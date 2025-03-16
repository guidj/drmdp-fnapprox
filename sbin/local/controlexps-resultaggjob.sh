#!/bin/bash
set -xe

DIR=$(dirname $0)
PARENT_DIR=$DIR/../..
BASE=drmdp

DATA_DIR=$1
INPUT_DIR=$HOME/fs/$BASE/workflows/controljob/logs/${DATA_DIR}
OUTPUT_DIR=$HOME/fs/$BASE/workflows/controljob/agg/${DATA_DIR}/`date +%s`

rm -rf ${OUTPUT_DIR}
python $PARENT_DIR/src/$BASE/workflows/control_results_agg_pipeline.py \
    --input-dir=$INPUT_DIR \
    --output-dir=$OUTPUT_DIR 