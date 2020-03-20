all: stickman.o python


stickman:stickman.o
	nvcc -o $@ $^ -O3 -std=c++11 --use_fast_math -Xcompiler -fPIC `pkg-config --libs opencv`


stickman.o: stickman.cpp
	nvcc -o $@ $^ -O3 -c -std=c++11 --use_fast_math -Xcompiler -fPIC

python: stickman.o
	swig -Wall -c++ -python stickman.i
	g++ -fPIC -O3 -ffast-math -std=c++11 -Wall -Wextra -shared stickman_wrap.cxx -o _stickman.so -L. stickman.o -I/opt/conda/include/python3.6m -I/opt/conda/lib/python3.6/site-packages/numpy/core/include -L/opt/conda/lib -lpython3.6m


clean:
	rm -rf keypoints2img *.o *.cxx *.so *.py


