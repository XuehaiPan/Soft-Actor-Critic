#!/usr/bin/env bash

ENV="Pendulum-v0"

ROOT_DIR="$(
	cd "$(dirname "$(dirname "$0")")"
	pwd
)"

cd "$ROOT_DIR"

PYTHONWARNINGS=ignore python3 main.py \
	--mode test --gpu 0 1 2 3 --env "$ENV" \
	--hidden-dims 128 64 \
	--activation LeakyReLU \
	--encoder-arch RNN \
	--state-dim 128 \
	--encoder-hidden-dims-before-lstm 128 \
	--encoder-hidden-dims-lstm 64 \
	--encoder-hidden-dims-after-lstm 128 \
	--skip-connection \
	--n-episodes 100 \
	--n-samplers 4 \
	--random-seed 0 \
	--log-dir "logs/$ENV/RNN" \
	--checkpoint-dir "checkpoints/$ENV/RNN" \
	--load-checkpoint
