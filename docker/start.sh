#!/bin/bash
SCRIPT_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
containerId=dreamax:v1
xhost +local:`docker inspect --format='{{ .Config.Hostname }}' $containerId` # opening xhost locally, see http://wiki.ros.org/docker/Tutorials/GUI 
docker run \
    -it \
    --rm \
    --net=host \
    --name=jax \
    --privileged \
    -e host_uid=$(id -u) \
    -e host_gid=$(id -g) \
    -v $SCRIPT_DIR/..:/dreamax \
    --gpus all \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    -w="/dreamax" \
    --runtime nvidia \
    $containerId \
    bash
