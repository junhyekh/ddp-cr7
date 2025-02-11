#!/usr/bin/env bash

set -exu

# Figure out repository root.
SCRIPT_DIR="$( cd "$( dirname $(realpath "${BASH_SOURCE[0]}") )" && pwd )"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"

# CACHE_PATH="/home/${USER}/.cache/ddp"
IMAGE_TAG='hhjjhh0425/ddp:test'


# Launch docker with the following configuration:
# * Display/Gui connected
# * Network enabled (passthrough to host)
# * Privileged
# * GPU devices visible
# * Current working git repository mounted at ${HOME}
# * 8Gb Shared Memory
docker run -it \
    -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
    --mount type=bind,source="${REPO_ROOT}",target="/root/$(basename ${REPO_ROOT})" \
    --shm-size=32g \
    --network host \
    --privileged \
    --gpus all \
    "$@" \
    "${IMAGE_TAG}"
