#!/usr/bin/env bash

ENV="InvertedPendulumBulletEnv-v0"
DATETIME="$(date +"%Y-%m-%d-%T")"
LOG_DIR="logs/$ENV/CNN/$DATETIME"
CHECKPOINT_DIR="checkpoints/$ENV/CNN"

ROOT_DIR="$(
	cd "$(dirname "$(dirname "$0")")"
	pwd
)"

cd "$ROOT_DIR"
mkdir -p "$LOG_DIR"
cp "$0" "$LOG_DIR"

PYTHONWARNINGS=ignore python3 main.py \
	--mode test --gpu 0 1 0 1 \
	--env "$ENV" \
	--vision-observation --image-size 128 \
	--n-frames 3 \
	--hidden-dims 128 64 \
	--activation LeakyReLU \
	--encoder-arch CNN \
	--state-dim 128 \
	--encoder-hidden-channels 64 64 64 64 \
	--kernel-sizes 5 5 3 3 \
	--strides 1 1 1 1 \
	--paddings 2 2 1 1 \
	--poolings 2 2 2 1 \
	--encoder-activation LeakyReLU \
	--n-episodes 100 \
	--n-samplers 4 \
	--random-seed 0 \
	--log-episode-video \
	--log-dir "$LOG_DIR" \
	--checkpoint-dir "$CHECKPOINT_DIR" \
	--load-checkpoint \
	"$@" # script arguments (can override args above)
