cnn: cnn.cc
	g++ -Wall -I../../include -I../../build/include $^ -o $@ -L../../build/lib -L/usr/local/lib -lsinga -lprotobuf -lglog

cnn-cuda: cnn.cc
	g++ -Wall -I../../include -I../../build/include -I/usr/local/cuda/include $^ -o $@ -L../../build/lib -L/usr/local/lib -lsinga -lprotobuf -lglog

clean:
	rm -f cnn cnn-cuda
