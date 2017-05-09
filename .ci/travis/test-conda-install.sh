#! /bin/bash

conda create -n _conda_install python=$TRAVIS_PYTHON_VERSION numpy=$NUMPY_VERSION
source activate _conda_install
conda install --file=requirements.txt
python setup.py develop
cd scripts && ./test-installed-landlab.py
source deactivate
conda remove -n _conda_install --all
