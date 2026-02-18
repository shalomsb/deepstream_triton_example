#!/bin/bash
set -e
cd "${0%/*}"
SETUP_SCRIPT_DIR=$PWD
PYDS_WHL_FILE_NAME=pyds-1.2.0-cp310-cp310-linux_$(arch).whl
PERSISTENT_BUILD_PATH=${SETUP_SCRIPT_DIR}/wheels

if [ -e ${PERSISTENT_BUILD_PATH}/${PYDS_WHL_FILE_NAME} ]; then
    echo "Bindings already exist! Skipping installation."
    exit;
fi

if [ ! -e "/opt/nvidia/deepstream/deepstream-7.1/sources/deepstream_python_apps/" ]; then
    bash /opt/nvidia/deepstream/deepstream-7.1/user_deepstream_python_apps_install.sh -v 1.2.0;
fi

echo "Installing build dependencies"
python3 -m pip install -q build

echo "Uninstalling existing pyds installation"
python3 -m pip uninstall -y pyds

cd /opt/nvidia/deepstream/deepstream-7.1/sources/deepstream_python_apps/
git checkout v1.2.0
git submodule update --init

echo "Copying custom binding files..."
cp ${SETUP_SCRIPT_DIR}/bindnvdsmeta.cpp /opt/nvidia/deepstream/deepstream-7.1/sources/deepstream_python_apps/bindings/src/.
cp ${SETUP_SCRIPT_DIR}/bindtrackermeta.cpp /opt/nvidia/deepstream/deepstream-7.1/sources/deepstream_python_apps/bindings/src/.
cp ${SETUP_SCRIPT_DIR}/nvdsmetadoc.h /opt/nvidia/deepstream/deepstream-7.1/sources/deepstream_python_apps/bindings/docstrings/.
cp ${SETUP_SCRIPT_DIR}/trackermetadoc.h /opt/nvidia/deepstream/deepstream-7.1/sources/deepstream_python_apps/bindings/docstrings/.

cd bindings/

export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
echo "Building new PyDs bindings"
python3 -m build
cd dist/
pip3 install ${PYDS_WHL_FILE_NAME}

python3 - <<EOF
import pyds
pyds.NVDS_TRACKER_SHADOW_LIST_META
EOF

if [ $? -eq 0 ]; then
    echo "Custom PYDS bindings installation success";
else
    echo "Error installing custom PYDS bindings";
fi

mkdir -p ${PERSISTENT_BUILD_PATH}
cp ${PYDS_WHL_FILE_NAME}  ${PERSISTENT_BUILD_PATH}/${PYDS_WHL_FILE_NAME};
