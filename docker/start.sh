#!/bin/bash
SCRIPT_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
containerId=dreamax:v3
xhost +local:`docker inspect --format='{{ .Config.Hostname }}' $containerId` # opening xhost locally, see http://wiki.ros.org/docker/Tutorials/GUI

# define array for docker run call, allows to comment individual arguments
run_args=(
    run
    -it                         # interactive, allocate a pseudo-TTY
    --rm                        # automatically remove the container when it exits
    -e host_uid=$(id -u)        # env var to host user's id
    -e host_gid=$(id -g)        # env var with host user's group id

    # optionally pass through host user
    #--user=$(id -u $USER):$(id -g $USER)
    #--volume="/etc/group:/etc/group:ro"
    #--volume="/etc/passwd:/etc/passwd:ro"
    #--volume="/etc/shadow:/etc/shadow:ro"
    #--volume="/etc/sudoers.d:/etc/sudoers.d:ro"

    --net=host                  # use host network
    --name=dreamax              # name of container
    -v $SCRIPT_DIR/..:/dreamax  # mount source code directory to /dreamer

    # To restrict GPU availability inside the docker container (e.g. to hide your display GPU) you can use:
    # --gpus '"device=1,2,3"'
    # see https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html
    --gpus all                  # specify which GPUs to use

    -v /tmp/.X11-unix:/tmp/.X11-unix    # expose UNIX-domain socket to communicate with the X server
    -e display                  # pass through DISPLAY environment variable
    -w /dreamax                 # set working directory for bash to /dreamax
    --runtime nvidia            # use nvidia runtime
    $containerId
    bash                        # command to execute
)

docker ${run_args[@]}
