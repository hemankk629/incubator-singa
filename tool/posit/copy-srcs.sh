#!/bin/bash

cd ../../
mkdir -p build_posit
cd build_posit

cp ../tool/posit/Makefile .

cp ../examples/cpp/cifar10/cifar10.h .
cp ../examples/cpp/cifar10/cnn.cc .
cp ../src/core/tensor/*.cc .
cp ../src/core/tensor/*.h .
rm tensor_math_cuda.h
rm tensor_math_opencl.h
cp ../src/core/device/cpp_cpu.cc .
cp ../src/core/device/device.cc .
cp ../src/core/scheduler/scheduler.cc .
cp ../src/core/common/common.cc .
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

rm cudnn_*
rm opencl_*
rm *.cl
rm *.cu
