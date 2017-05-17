#! /bin/bash

# conda create -n _pip_install python=$TRAVIS_PYTHON_VERSION
# source activate _pip_install
# pip install -r requirements.txt
# if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
#   conda install python.app
#   PYTHON=pythonw
# else
#   PYTHON=python
# fi
# PYTHON=python
# $PYTHON setup.py develop
# cd scripts && $PYTHON ./test-installed-landlab.py || exit 1
# source deactivate
# conda remove -n _pip_install --all

mkdir _testing
cd _testing
pip install landlab
python -c 'import landlab; landlab.test()'
