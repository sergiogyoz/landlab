#! /bin/bash

conda create -n _conda_install python=$TRAVIS_PYTHON_VERSION
source activate _conda_install
conda install --file=requirements.txt
conda install 'hdf4>=4.2.12'
python setup.py develop
cd scripts && ./test-installed-landlab.py || exit 1
source deactivate
conda remove -n _conda_install --all
