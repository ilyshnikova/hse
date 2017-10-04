all: run

build:
	 g++ main.cpp -O2 -std=c++1z -pthread -g -o main

run: build
	 ./main seq-read && rm _tmp_file_
	 ./main seq-write && rm _tmp_file_
	 ./main rnd-read && rm _tmp_file_
	 ./main rnd-write && rm _tmp_file_
	 ./main rnd-read-parallel && rm _tmp_file_*
	 ./main rnd-write-parallel && rm _tmp_file_*
	 ./main rnd-mixed-parallel && rm _tmp_file_*
