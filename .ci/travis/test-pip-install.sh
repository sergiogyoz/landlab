#! /bin/bash


install_landlab() {
  conda create -n _pip_install python=$1
  source activate _pip_install

  if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
    conda install python.app
    PYTHON=pythonw
  else
    PYTHON=python
  fi

  mkdir -p _testing
  cd _testing

  pip install -r requirements.txt
  pip install landlab

  $PYTHON -c 'import landlab; landlab.test()' || exit -1

  source deactivate
  conda remove -n _pip_install --all
}


PYTHON_VERSIONS=$*

for v in $PYTHON_VERSIONS; do
  install_landlab $v || exit -1
done
