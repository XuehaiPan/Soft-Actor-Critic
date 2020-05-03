#!/usr/bin/env bash

ENV="InvertedPendulumBulletEnv-v0"

ROOT_DIR="$(
	cd "$(dirname "$(dirname "$0")")"
	pwd
)"

cd "$ROOT_DIR"

PYTHONWARNINGS=ignore python3 main.py \
	--mode test --gpu 0 1 2 3 --env "$ENV" \
	--vision-observation --image-size 128 \
	--hidden-dims 128 64 \
	--activation LeakyReLU \
	--encoder-arch CNN \
	--state-dim 128 \
	--encoder-hidden-channels 64 64 64 \
	--kernel-sizes 3 3 3 \
	--strides 1 1 1 \
	--paddings 1 1 1 \
	--n-episodes 100 \
	--n-samplers 4 \
	--random-seed 0 \
	--log-dir "logs/$ENV/FC" \
	--checkpoint-dir "checkpoints/$ENV/FC" \
	--load-checkpoint
