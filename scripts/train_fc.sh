#!/usr/bin/env bash

ENV="BipedalWalker-v3"

ROOT_DIR="$(
	cd "$(dirname "$(dirname "$0")")"
	pwd
)"

cd "$ROOT_DIR"

python3 main.py --mode train --gpu 0 --env "$ENV" \
	--net FC --activation LeakyReLU \
	--hidden-dims 256 256 256 128 128 128 \
	--max-episodes 4000 --max-episode-steps 500 \
	--n-updates 16 --batch-size 256 \
	--buffer-capacity 1000000 --lr 1E-4 --weight-decay 1E-5 --random-seed 0 \
	--log-dir "logs/$ENV/FC" \
	--checkpoint-dir "checkpoints/$ENV/FC"
