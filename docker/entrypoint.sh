#!/usr/bin/env bash

function setup_pyds_bindings
{
    PYDS_WHL_FILE_NAME=pyds-1.2.2-cp312-cp312-linux_$(arch).whl
    # Build the python bindings if they do not exist
    if [ ! -e "/opt/scripts/patch_pyds_shadow_tracker_bindings/wheels/${PYDS_WHL_FILE_NAME}" ]; then
        cd /opt/scripts/patch_pyds_shadow_tracker_bindings
        ./install_custom_bindings.sh
    fi
    # Force-Install updated pyds with shadow tracker patch
    python3 -m pip install /opt/scripts/patch_pyds_shadow_tracker_bindings/wheels/${PYDS_WHL_FILE_NAME} --force-reinstall > /dev/null 2>&1
}

function usage
{
    echo "usage: ./docker_entrypoint.sh [-r/-d/-s]"
    echo "Choose action from:"
    echo "  -r | Run deepstream app"
    echo "  -s | Run Triton server"
    echo "  -d | develop using bash terminal"
    echo "  -h | Help"
}

ACTION=""

# Make sure we have 1 input argument to the script
if [[ $# -ne 1 ]]; then
    usage && exit;
fi

# Read command line arguments
while [[ "$1" != "" ]]; do
    case $1 in
        -b | -r | -d | -s )          ACTION=$1    ;;
        -h )                    usage && exit;;
        * )                     usage && exit;;
    esac
    shift;
done

if [[ $ACTION == '-b' ]]; then
    echo "Building custom pyds bindings..."
    cd /opt/scripts/patch_pyds_shadow_tracker_bindings
    ./install_custom_bindings.sh
    bash /opt/scripts/install_models.sh
elif [[ $ACTION == '-r' ]]; then
    setup_pyds_bindings
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
    setup_pyds_bindings
    /bin/bash
else
    usage && exit;
fi

