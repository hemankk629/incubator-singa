#!/bin/bash

DST="build_posit"

cd ../../
rm -rf $DST
mkdir $DST
cd $DST

cp ../tool/posit/Makefile .

cp ../examples/cifar10/cifar10.h .
cp ../examples/cifar10/cnn.cc .
cp ../src/core/tensor/* .
cp ../src/core/device/device.cc .
cp ../src/core/device/platform.cc .
cp ../src/core/device/cpp_cpu.cc .
cp ../src/io/binfile* .
cp ../src/io/snapshot.cc .
cp ../src/io/textfile_writer.cc .
cp ../src/model/layer/* .
cp ../src/model/optimizer/* .
cp ../src/model/loss/* .
cp ../src/model/metric/* .
cp ../src/model/updater/* .
cp ../src/utils/* .
cp ../src/model/feed_forward_net.cc .
cp ../build/src/*.h .
cp ../build/src/*.cc .
