#!/bin/bash
cd "${0%/*}/.."

function usage 
{
    echo "usage: ./launch.sh [-b/-d/-r]"
    echo "Choose action from:"
    echo "      -b | Build Docker container and setup environment"
    echo "      -d | Develop Inside Docker container "
    echo "      -r | Run deepstream application"
}

# Initialize parameters with default values
ACTION=""

# Validate we have exactly 1 command arguments
if [[ $# -ne 1 ]]; then
	usage && exit;
fi

while [[ "$1" != "" ]]; do
    case $1 in
        -b | -d | -r  | -s)
                        ACTION=$1
                        ;;
        -h )
                        usage
                        exit
                        ;;
        * ) 
                        usage
                        exit
                        ;;
    esac
    shift;
done


# Validate we read both parameters
if [[ $ACTION == "" ]]; then
	usage && exit;
fi

BASE_IMAGE=nvcr.io/nvidia/deepstream:8.0-triton-multiarch
DOCKER_FILE=docker/Dockerfile
DOCKER_TAG=taodetr
DOCKER_TAG_VERSION=v1.0
DOCKER_NAME=taodetr

# # Set container name based on action flag
# if [[ $ACTION == '-s' ]]; then
#     CONTAINER_NAME="ds8.0-s"
# elif [[ $ACTION == '-d' ]]; then
#     CONTAINER_NAME="ds8.0-d"
# else
#     CONTAINER_NAME="ds8.0"
# fi

if ! [[ -z $DISPLAY ]]; then
    xhost +local:root
fi

if [[ $ACTION == '-b' ]]; then
    time \
    docker build -f $DOCKER_FILE -t $DOCKER_TAG:$DOCKER_TAG_VERSION . \
        --build-arg BASE_IMAGE=${BASE_IMAGE};
    
    # We do this to build the TensorRT engine and download models into the mounted volume
	docker run --name ${DOCKER_NAME} \
    	--rm --net=host --ipc=host --shm-size=4g --privileged -it \
		--runtime=nvidia --gpus all \
		-v "$(pwd)/deepstream":/deepstream \
		-v "$(pwd)/triton/model_repo":/triton/model_repo \
        -v "$(pwd)/scripts":/opt/scripts \
		--entrypoint /opt/entrypoint.sh \
		$DOCKER_TAG:$DOCKER_TAG_VERSION \
		$ACTION;
    exit;

elif [[ $ACTION == '-r' ]] || [[ $ACTION == '-s' ]] || [[ $ACTION == '-d' ]] ; then
    # Run Triton server
    docker run --name=${DOCKER_NAME} --rm \
        -e DISPLAY=$DISPLAY \
        --rm --net=host --ipc=host --shm-size=4g --privileged -it \
		--runtime=nvidia --gpus all \
		-v "$(pwd)/deepstream":/deepstream \
		-v "$(pwd)/triton/model_repo":/triton/model_repo \
        -v "$(pwd)/scripts":/opt/scripts \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v /dev:/dev \
		--entrypoint /opt/entrypoint.sh \
		$DOCKER_TAG:$DOCKER_TAG_VERSION \
		$ACTION;
    exit;

else
    echo "Invalid option from [action <-b/-r/-d/-s>]"
    usage
    exit 1
fi