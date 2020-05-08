#!/bin/bash

set -x

SRC_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd $SRC_DIR

cd ../../../

INCLUDE1=`pwd`/include
INCLUDE2=`pwd`/build/include
LIB1=`pwd`/build/lib

cd $SRC_DIR

g++ -I$INCLUDE1 -I$INCLUDE2 cnn.cc -o cnn -L$LIB1 -lsinga -lglog
