#!/usr/bin/env bash

function usage
{
    echo "usage: ./entrypoint.sh [-r/-d/-s/-b]"
    echo "  -r | Run deepstream app"
    echo "  -s | Run Triton server"
    echo "  -d | Develop using bash terminal"
    echo "  -b | Build models"
    echo "  -h | Help"
}

ACTION=""

if [[ $# -ne 1 ]]; then
    usage && exit;
fi

while [[ "$1" != "" ]]; do
    case $1 in
        -b | -r | -d | -s )    ACTION=$1    ;;
        -h )                    usage && exit;;
        * )                     usage && exit;;
    esac
    shift;
done

if [[ $ACTION == '-b' ]]; then
    bash /opt/scripts/install_models.sh
elif [[ $ACTION == '-r' ]]; then
    cd /deepstream
    python3 main.py
elif [[ $ACTION == '-s' ]]; then
    tritonserver --model-repository=/triton/model_repo \
                --allow-http=true \
                --allow-grpc=true \
                --http-port=8000 \
                --grpc-port=8001 \
                --metrics-port=8002
elif [[ $ACTION == '-d' ]]; then
    /bin/bash
else
    usage && exit;
fi
