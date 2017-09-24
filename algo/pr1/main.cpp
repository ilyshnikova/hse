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
#include <sys/disk.h>

void seq_write(const size_t blockSize = 1/*in MiB*/, const size_t count=100) {
    // open fd
    int fd = open("file", O_RDWR | O_CREAT | O_SYNC, 0644);

    // create buffer with random data
    char* buffer = new char[blockSize];
    std::mt19937 gen{ std::random_device()() };
    std::uniform_int_distribution<> dis(0, 255);
    std::generate_n(buffer, blockSize * 1024 * 1024, [&]{ return dis(gen); });
    buffer[blockSize * 1024 * 1024 - 1] = '\0';

    // start timer
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

//    lseek(fd, 0, SEEK_END);

    for (size_t i = 0; i < count; ++i) {
        write(fd, buffer, blockSize * 1024 * 1024);
    }

    // finish timer
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    // compute time
    double time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

    // finalize
    delete[] buffer;
    close(fd);

    // print results
    std::cout << "speed: " << 1. * blockSize * count / time * 1000 * 1000 << std::endl;
}





class TProfiler {
protected:
    size_t IterationNumber;
    size_t BlockSize;
    int fd;
    char* Buffer;
public:
    TProfiler(const char mode, const size_t blockSize = 1/*MiB*/)
        : IterationNumber(100)
        , BlockSize(blockSize * 1024 * 1024)
        , Buffer(nullptr)
    {
        if (mode == 'w') {
            fd = open("file", O_RDWR | O_CREAT | O_SYNC, 0644);
            Buffer = new char[BlockSize];
            std::mt19937 gen{ std::random_device()() };
            std::uniform_int_distribution<> dis(0, 255);
            std::generate_n(Buffer, BlockSize, [&]{ return dis(gen); });
            Buffer[BlockSize - 1] = '\0';

            // for linux
            // fd = open(name, O_RDWR | O_CREAT | O_SYNC | O_DIRECT, 0644);
        } else if (mode == 'r') {
            fd = open("file", O_RDONLY);
        }
    }

    TProfiler(const size_t iterationNumber, const size_t actionsNumber)
        : IterationNumber(iterationNumber)
        , BlockSize(1024)
    {}

    virtual void Action() = 0;

    virtual void Stop() {};

    double Lanch()  {
        double resultTime = 0;
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        Action();
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        return  static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    }

    std::tuple<double, double> Compute() {
        std::vector<double> results(IterationNumber);
        std::generate_n(results.begin(), IterationNumber, [&]{ return Lanch(); });

        double mean = std::accumulate(results.begin(), results.end(), 0.0,
                                                         [&](double a, double b){ return a + b / IterationNumber; });

        double var = std::accumulate(results.begin(), results.end(), 0.0,
                                                        [&](double a, double b) { return a + (b - mean) * (b - mean) / IterationNumber; });

        std::cout << "mean: " << mean << " " << BlockSize << " bytes "  << IterationNumber << " times" << std::endl;
        std::cout << "speed: " << static_cast<double>(BlockSize) / 1024 / 1024 / (mean / 1000 / 1000) << std::endl;

        return {mean, std::sqrt(var)};
    }

    virtual ~TProfiler() {
        if (Buffer != nullptr) {
            delete[] Buffer;
        }
        close(fd);
    }
};

class TSeqWriterProfiler : public TProfiler {
private:
    char* Buf;
public:

    TSeqWriterProfiler()
        : TProfiler('w')
    {}

    void Action() {
        lseek(fd, 0, SEEK_END);
    }

    ~TSeqWriterProfiler() {
        delete[] Buf;
        close(fd);
    }

};

void create_file(const size_t mb_size) {
    std::mt19937 gen{ std::random_device()() };
    std::uniform_int_distribution<> dis(0, 255);
    std::ofstream file("text_file.txt");
    std::generate_n(std::ostream_iterator<char>(file, ""), mb_size * 1024 * 1024, [&]{ return dis(gen); });
    file.close();
}

int main(int argc, char** argv) {
    //create_file(1); // create 1MB file

    seq_write();

//    TSeqWriterProfiler profiler;
//    double mean, var;
//
//    std::tie(mean, var) = profiler.Compute();
//    std::cout << mean << " " << var << std::endl;
//    std::cout << 1. / mean  * 1000 * 1000 * 1024 << std::endl;

//    if (argc < 2) {
//        std::cerr << "Incorrect number of arguments: " << argc << ", expected >=2" << std::endl;
//        return 1;
//    }
//
//    std::string mode = argv[1];
//    if (mode == "seq-read") {
//        std::cerr << "58 MB/s\n";
//    } else if (mode == "seq-write") {
//        std::cerr << "68 MB/s\n";
//    } else if (mode == "rnd-read") {
//        std::cerr << "1643 mcs\n";
//    } else if (mode == "rnd-write") {
//        std::cerr << "2743 mcs\n";
//    } else if (mode == "rnd-read-parallel") {
//        std::cerr << "1382 mcs\n";
//    } else if (mode == "rnd-write-parallel") {
//        std::cerr << "1402 mcs\n";
//    } else if (mode == "rnd-mixed-parallel") {
//        std::cerr << "19320 mcs\n";
//    } else {
//        std::cerr << "Incorrect mode " << mode << std::endl;
//        return 1;
//    }

    return 0;
}

