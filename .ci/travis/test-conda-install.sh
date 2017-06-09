#! /bin/bash

install_landlab() {
  conda create -n _conda_install python=$1
  source activate _conda_install

  mkdir -p _testing
  cd _testing

  conda install landlab -c landlab

  python -c 'import landlab; landlab.test()' || exit -1

  source deactivate
  conda remove -n _conda_install --all
}


PYTHON_VERSIONS=$*

for v in $PYTHON_VERSIONS; do
  install_landlab $v || exit -1
done
