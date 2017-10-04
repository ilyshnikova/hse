#include <iostream>
#include <string>
#include <random>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <iomanip>
#include <ctime>
#include <chrono>
#include <sys/ioctl.h>
#include <atomic>
#include <thread>

// remove for mac
#include <linux/fs.h>

const char* tmp_file_name = "_tmp_file_";

/*
 nastyats@bs-dev12:~$ sudo fdisk -l | grep "Sector size"
 Sector size (logical/physical): 512 bytes / 4096 bytes
*/

int open(const char* file_name, const char mode) {
    int fd;
    if (mode == 'w') {
        fd = open(file_name, O_WRONLY | O_CREAT | O_TRUNC | O_SYNC, 0644);
    } else if (mode == 'r') {
        fd = open(file_name, O_RDONLY);
    } else {
        std::cerr << "open function: Incorrct open mode" << std::endl;
        std::abort();
    }
    if (fd == -1) {
        std::cerr << "Can't open file " << file_name << std::endl;
        std::abort();
    }
    return fd;
}

template
<typename T>
T* create_buffer(const size_t buffer_size/*Kib*/, const bool empty = false, const int from = 0, const int to = 255) {
    const size_t size = buffer_size * 1024;
    T* buffer = new T[size];
    if (!empty) {
        std::mt19937 gen{ std::random_device()() };
        std::uniform_int_distribution<> dis(from, to);
        std::generate_n(buffer, size, [&]{return dis(gen);});
    }
    return buffer;
}

void create_file(const size_t block_size /*in KiB*/, const size_t count, const std::string& name) {
    int fd = open(name.c_str(), 'w');
    char* buffer = create_buffer<char>(block_size);
    for (size_t i = 0; i < count; ++i) {
        if (write(fd, buffer, block_size * 1024) == -1) {
            std::cerr << "While creating file " << name << " write failed" << std::endl;
            std::abort();
        }
    }
    delete[] buffer;
    close(fd);
}

template <typename Function>
double measure_time(Function func, const size_t count) {
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < count; ++i) {
        func(i);
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    return static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
}

void mac_drop_cache() {
    system("sudo purge");
}

double seq_write(const size_t block_size = 1024/*KiB*/, const size_t count=1024) {
    int fd = open(tmp_file_name, 'w');
    char* buffer = create_buffer<char>(block_size);

	double time = measure_time([&] (size_t) {
        if (write(fd, buffer, block_size * 1024) == -1) {
            std::cerr << "While seq_write write failed" << std::endl;
            std::abort();
        }
        fsync(fd);
	// remove for mac
        fdatasync(fd);
	}, count);

    delete[] buffer;
    close(fd);

    return  1. * block_size / 1024 * count / time * 1000 * 1000;
}

double seq_read(const size_t block_size = 1024/*KiB*/, const size_t count=1024) {
    create_file(block_size, count, tmp_file_name);
    int fd = open(tmp_file_name, 'r');

    // remove for mac
    if (posix_fadvise(fd, 0, block_size * 1024 * count, POSIX_FADV_DONTNEED) == -1) {
        std::cerr << "could not flush the cache" << std::endl;
    }
    // use thuis way for mac
    // mac_drop_cache();

	char* buffer = create_buffer<char>(block_size, true);
	double time = measure_time([&] (size_t) {
        if (read(fd, buffer, block_size * 1024) == -1) {
            std::cerr << "While seq_read read failed" << std::endl;
            std::abort();
        }
	}, count);

	close(fd);
    delete[] buffer;

    return 1. * block_size / 1024 * count / time * 1000 * 1000;
}

double rnd_write(const std::string& file_name = tmp_file_name, const size_t count=1024) {
    create_file(1024, count, tmp_file_name);
    int fd = open(tmp_file_name, 'w');

	const char* buffer = "a";
    int* write_locations = create_buffer<int>(count, false, 0, 1024 * 1024 * count - 10);

	double time = measure_time([&] (size_t i) {
		int to = write_locations[i];
		lseek(fd, to, SEEK_SET);
		if (write(fd, buffer, 1) == -1) {
            std::cerr << "While seq_write write failed" << std::endl;
            std::abort();
        }
        fsync(fd);
	// remove for mac
        fdatasync(fd);
	}, count);

	close(fd);
    return  time / count;
}

double rnd_read(const std::string& file_name = tmp_file_name, const size_t count=1024) {
    create_file(1024, count, file_name);
    int fd = open(tmp_file_name, 'r');

    // remove for mac
    if (posix_fadvise(fd, 0, 1024 * 1024 * count, POSIX_FADV_DONTNEED) == -1) {
        std::cerr << "could not flush the cache" << std::endl;
    }
    // use thuis way for mac
    // mac_drop_cache();

    char* buffer = new char[1];
    int* write_locations = create_buffer<int>(count, false, 0, 1024 * 1024 * count - 10);

	double time = measure_time([&] (size_t i) {
		int to = write_locations[i];
		lseek(fd, to, SEEK_SET);
        if (read(fd, buffer, 1) == -1) {
            std::cerr << "While seq_read read failed" << std::endl;
            std::abort();
        }
	}, count);

	close(fd);
	delete[] buffer;
    return time / count;
}

double rnd_mixed_parallel(const size_t read_threads_number, const size_t write_threads_number, const size_t count = 1024) {
    std::vector<std::thread> threads(read_threads_number + write_threads_number);
    std::atomic<int> result_time(0);
    for (size_t i = 0; i < read_threads_number + write_threads_number; ++i) {
        threads[i] = std::thread([&](){
            result_time += int(
                (i < read_threads_number)
                    ? rnd_read(std::string(tmp_file_name) + std::to_string(i), count)
                    : rnd_write(std::string(tmp_file_name) + std::to_string(i), count)
            );
        });
    }

    for (size_t i = 0; i < read_threads_number + write_threads_number; ++i) {
        threads[i].join();
    }

    return result_time / (read_threads_number + write_threads_number);
}


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Incorrect number of arguments: " << argc << ", expected >=2" << std::endl;
        return 1;
    }

    std::string mode = argv[1];
    if (mode == "seq-read") {
        std::cerr << seq_read() << " MB/s\n";
    } else if (mode == "seq-write") {
        std::cerr << seq_write() << " MB/s\n";
    } else if (mode == "rnd-read") {
        std::cerr << rnd_read() << " mcs\n";
    } else if (mode == "rnd-write") {
        std::cerr << rnd_write() << " mcs\n";
    } else if (mode == "rnd-read-parallel") {
        std::cerr << rnd_mixed_parallel(8, 0) << " mcs\n";
    } else if (mode == "rnd-write-parallel") {
        std::cerr << rnd_mixed_parallel(0, 8) << " mcs\n";
    } else if (mode == "rnd-mixed-parallel") {
        std::cerr << rnd_mixed_parallel(4, 4) << " mcs\n";
    } else {
        std::cerr << "Incorrect mode " << mode << std::endl;
        return 1;
    }

    return 0;
}
