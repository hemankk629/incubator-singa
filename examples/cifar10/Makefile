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

split-dataset-images: split-dataset-images.c
	gcc -Wall $^ -o $@

clean:
	rm -f cnn cnn-cuda
