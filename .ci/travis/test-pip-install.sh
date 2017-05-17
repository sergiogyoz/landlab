#! /bin/bash

PYTHON_VERSION=$1

conda create -n _pip_install python=$PYTHON_VERSION
source activate _pip_install
# pip install -r requirements.txt
if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
  conda install python.app
  PYTHON=pythonw
else
  PYTHON=python
fi

# $PYTHON setup.py develop
# cd scripts && $PYTHON ./test-installed-landlab.py || exit 1
# source deactivate
# conda remove -n _pip_install --all

mkdir -p _testing
cd _testing
pip install numpy
pip install landlab
$PYTHON -c 'import landlab; landlab.test()' || exit -1

source deactivate
conda remove -n _pip_install --all
