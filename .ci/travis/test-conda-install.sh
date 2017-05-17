#! /bin/bash

PYTHON_VERSION=$1

conda create -n _conda_install python=$PYTHON_VERSION
source activate _conda_install

# conda install --file=requirements.txt
# conda install 'hdf4>=4.2.12'
# python setup.py develop
# cd scripts && ./test-installed-landlab.py || exit 1
# source deactivate
# conda remove -n _conda_install --all

mkdir -p _testing
cd _testing
conda install landlab -c landlab
python -c 'import landlab; landlab.test()' || exit -1

source deactivate
conda remove -n _conda_install --all
