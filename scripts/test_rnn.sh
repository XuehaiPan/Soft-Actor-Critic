#!/usr/bin/env bash

ENV="BipedalWalkerHardcore-v3"

ROOT_DIR="$(
	cd "$(dirname "$(dirname "$0")")"
	pwd
)"

cd "$ROOT_DIR"

python3 main.py --mode test --gpu 0 --env "$ENV" \
	--net RNN --activation LeakyReLU \
	--hidden-dims-before-lstm 256 256 256 128 128 \
	--hidden-dims-lstm 128 \
	--hidden-dims-after-lstm 128 128 \
	--skip-connection \
	--max-episodes 100 \
	--buffer-capacity 0 --random-seed 0 \
	--log-dir "logs/$ENV/RNN" \
	--checkpoint-dir "checkpoints/$ENV/RNN"
