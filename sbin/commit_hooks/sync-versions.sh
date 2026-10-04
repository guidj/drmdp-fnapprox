#!/bin/bash
set -e

uv sync --default-index https://pypi.org/simple
git add -u uv.lock
