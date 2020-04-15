#!/usr/bin/env bash

ENV="Pendulum-v0"

ROOT_DIR="$(
	cd "$(dirname "$(dirname "$0")")"
	pwd
)"

cd "$ROOT_DIR"

PYTHONWARNINGS=ignore python3 main.py \
	--mode test --gpu 0 1 2 3 --env "$ENV" --render \
	--net RNN --activation LeakyReLU \
	--hidden-dims-before-lstm 128 \
	--hidden-dims-lstm 128 \
	--hidden-dims-after-lstm 64 \
	--skip-connection \
	--state-dim 128 \
	--encoder-hidden-dims 128 128 128 128 \
	--n-episodes 100 \
	--n-samplers 4 \
	--random-seed 0 \
	--log-dir "logs/$ENV/RNN" \
	--checkpoint-dir "checkpoints/$ENV/RNN" \
	--load-checkpoint
