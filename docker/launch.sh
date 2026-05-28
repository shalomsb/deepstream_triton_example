#!/bin/bash
cd "${0%/*}/.."

function usage
{
    echo "usage: ./launch.sh [-b/-d/-r/-s]"
    echo "      -b | Build Docker container and setup models"
    echo "      -d | Develop inside Docker container"
    echo "      -r | Run deepstream application"
    echo "      -s | Run Triton server"
}

ACTION=""

if [[ $# -ne 1 ]]; then
	usage && exit;
fi

while [[ "$1" != "" ]]; do
    case $1 in
        -b | -d | -r | -s )    ACTION=$1 ;;
        -h )                    usage && exit ;;
        * )                     usage && exit ;;
    esac
    shift;
done

if [[ $ACTION == "" ]]; then
	usage && exit;
fi

BASE_IMAGE=nvcr.io/nvidia/deepstream:9.0-triton-multiarch
DOCKER_FILE=docker/Dockerfile
DOCKER_TAG=yolo26x-ds
DOCKER_TAG_VERSION=v1.0
DOCKER_NAME=yolo26x-ds

if ! [[ -z $DISPLAY ]]; then
    xhost +local:root
fi

if [[ $ACTION == '-b' ]]; then
    time \
    docker build -f $DOCKER_FILE -t $DOCKER_TAG:$DOCKER_TAG_VERSION . \
        --build-arg BASE_IMAGE=${BASE_IMAGE};

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
    docker run --name=${DOCKER_NAME} --rm \
        -e DISPLAY=$DISPLAY \
        --net=host --ipc=host --shm-size=4g --privileged -it \
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
    echo "Invalid option"
    usage
    exit 1
fi
