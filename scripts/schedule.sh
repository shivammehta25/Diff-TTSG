#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python diff_ttsg/train.py trainer.max_epochs=5 logger=csv

python diff_ttsg/train.py trainer.max_epochs=10 logger=csv
