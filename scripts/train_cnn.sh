#!/usr/bin/env bash

ENV="InvertedPendulumBulletEnv-v0"
LOG_DIR="logs/$ENV/CNN"
CHECKPOINT_DIR="checkpoints/$ENV/CNN"

ROOT_DIR="$(
	cd "$(dirname "$(dirname "$0")")"
	pwd
)"

cd "$ROOT_DIR"
mkdir -p "$LOG_DIR"
cp "$0" "$LOG_DIR"

PYTHONWARNINGS=ignore python3 main.py \
	--mode train --gpu 0 1 2 3 4 \
	--env "$ENV" \
	--vision-observation --image-size 128 \
	--hidden-dims 128 64 \
	--activation LeakyReLU \
	--encoder-arch CNN \
	--state-dim 128 \
	--encoder-hidden-channels 64 64 64 \
	--kernel-sizes 3 3 3 \
	--strides 1 1 1 \
	--paddings 1 1 1 \
	--max-episode-steps 500 \
	--n-epochs 1000 --n-updates 256 --batch-size 128 \
	--n-samplers 4 \
	--buffer-capacity 100000 \
	--update-sample-ratio 2.0 \
	--soft-q-lr 1E-3 --policy-lr 1E-4 \
	--alpha-lr 1E-3 --initial-alpha 1.0 --adaptive-entropy \
	--gamma 0.99 --soft-tau 0.01 \
	--normalize-rewards --reward-scale 1.0 \
	--weight-decay 1E-5 --random-seed 0 \
	--log-episode-video \
	--log-dir "$LOG_DIR" \
	--checkpoint-dir "$CHECKPOINT_DIR"