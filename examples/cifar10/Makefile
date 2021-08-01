cnn: cnn.cc
	g++ -std=c++11 -Wall -I../../include -I../../build/include $^ -o $@ -L../../build/lib -L/usr/local/lib -lsinga -lprotobuf -lglog

cnn-cuda: cnn.cc
	g++ -std=c++11 -Wall -I../../include -I../../build/include -I/usr/local/cuda/include $^ -o $@ -L../../build/lib -L/usr/local/lib -lsinga -lprotobuf -lglog

cnn-myfile: cnn.cc file_reader.cc
	g++ -std=c++11 -DMY_FILE_READER -Wall -I../../include -I../../build/include $^ -o $@ -L../../build/lib -L/usr/local/lib -lsinga -lprotobuf -lglog

cnn-cuda-myfile: cnn.cc file_reader.cc
	g++ -std=c++11 -DMY_FILE_READER -Wall -I../../include -I../../build/include -I/usr/local/cuda/include $^ -o $@ -L../../build/lib -L/usr/local/lib -lsinga -lprotobuf -lglog

cnn-bare: images.o labels.o mysnap.o mem_reader.cc cnn-bare.cc
	g++ -std=c++11 -Wall -g -I../../include -I../../build/include $^ -o $@ -L../../build/lib -L/usr/local/lib -lsinga -lprotobuf -lglog

cnn-bare-myfile: images.o labels.o myfilesnap.o mem_reader.cc cnn-bare.cc
	g++ -std=c++11 -DMY_FILE_READER -Wall -g -I../../include -I../../build/include $^ -o $@ -L../../build/lib -L/usr/local/lib -lsinga -lprotobuf -lglog

cnn-one-image: cnn-one-image.cc
	g++ -std=c++11 -DLOCAL_MAIN -Wall -g -I../../include -I../../build/include $^ -o $@ -L../../build/lib -L/usr/local/lib -lsinga -lprotobuf -lglog

cnn-one-image-cuda: cnn-one-image.cc
	g++ -std=c++11 -DLOCAL_MAIN -Wall -g -I../../include -I../../build/include -I/usr/local/cuda/include $^ -o $@ -L../../build/lib -L/usr/local/lib -lsinga -lprotobuf -lglog

cnn-hadoop-pipes: cnn-hadoop-pipes.cc cnn-one-image.cc
	g++ -std=c++11 -Wall -g -I../../include -I../../build/include -I/usr/local/cuda/include -I$(WORKING_DIR)/hadoop-2.10.1/include $^ -o $@ -L../../build/lib -L/usr/local/lib -L$(WORKING_DIR)/hadoop-2.10.1/lib/native -L$(JAVA_HOME)/jre/lib/amd64/server -lsinga -lprotobuf -lglog -lhdfs -lhadooppipes -lpthread -ljvm -lcrypto

split-dataset-images: split-dataset-images.c
	gcc -Wall $^ -o $@

lenet: lenet.cc
	g++ -std=c++11 -Wall -I../../include -I../../build/include $^ -o $@ -L../../build/lib -L/usr/local/lib -lsinga -lprotobuf -lglog

lenet-cuda: letnet.cc
	g++ -std=c++11 -Wall -I../../include -I../../build/include -I/usr/local/cuda/include $^ -o $@ -L../../build/lib -L/usr/local/lib -lsinga -lprotobuf -lglog

clean:
	rm -f cnn cnn-cuda
