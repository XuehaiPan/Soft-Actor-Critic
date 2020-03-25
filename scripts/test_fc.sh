#!/usr/bin/env bash

ENV="BipedalWalkerHardcore-v3"

ROOT_DIR="$(
	cd "$(dirname "$(dirname "$0")")"
	pwd
)"

cd "$ROOT_DIR"

python3 main.py --mode test --gpu 0 --env "$ENV" \
	--net FC --activation LeakyReLU \
	--hidden-dims 256 256 256 128 128 128 \
	--max-episodes 100 \
	--buffer-capacity 0 --random-seed 0 \
	--log-dir "logs/$ENV/FC" \
	--checkpoint-dir "checkpoints/$ENV/FC" \
    --load-checkpoint
