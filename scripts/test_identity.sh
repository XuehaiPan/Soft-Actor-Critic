#!/usr/bin/env bash

ENV="Pendulum-v0"
DATETIME="$(date +"%Y-%m-%d-%T")"
LOG_DIR="logs/$ENV/Identity/$DATETIME"
CHECKPOINT_DIR="checkpoints/$ENV/Identity"

ROOT_DIR="$(
	cd "$(dirname "$(dirname "$0")")"
	pwd
)"

cd "$ROOT_DIR"
mkdir -p "$LOG_DIR"
cp "$0" "$LOG_DIR"

PYTHONWARNINGS=ignore python3 main.py \
	--mode test --gpu 0 1 2 3 \
	--env "$ENV" \
	--hidden-dims 128 64 \
	--activation LeakyReLU \
	--n-episodes 100 \
	--n-samplers 4 \
	--random-seed 0 \
	--log-dir "$LOG_DIR" \
	--checkpoint-dir "$CHECKPOINT_DIR" \
	--load-checkpoint
