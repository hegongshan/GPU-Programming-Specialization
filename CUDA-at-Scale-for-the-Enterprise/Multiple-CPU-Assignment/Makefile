IDIR=./
COMPILER=nvcc
COMPILER_FLAGS=-I$(IDIR) -I/usr/local/cuda/include -lcuda --std c++17

.PHONY: clean build run

build: multi_cpu.cu multi_cpu.h
	$(COMPILER) $(COMPILER_FLAGS) multi_cpu.cu -o multi_cpu.exe -Wno-deprecated-gpu-targets

clean:
	rm -f multi_cpu.exe *.csv *.lock

run: run_producer_consumer_a run_producer_consumer_b run_multicpu_cuda

run_multicpu_cuda:
	# TODO: add an argument -- number of iterations
	./multi_cpu.exe 10

run_producer_consumer_a:
	# TODO: add an argument -- number of iterations
	python3 producer_consumer.py a 128 10

run_producer_consumer_b:
	# TODO: add an argument -- number of iterations
	python3 producer_consumer.py b 128 10

all: clean build run
