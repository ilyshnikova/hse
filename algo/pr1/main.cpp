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

class TProfiler {
protected:
    size_t IterationNumber;
    size_t ActionsNumber;
public:
    TProfiler()
        : IterationNumber(1000)
        , ActionsNumber(1024)
    {}

    TProfiler(const size_t iterationNumber, const size_t actionsNumber)
        : IterationNumber(iterationNumber)
        , ActionsNumber(actionsNumber)
    {}

    virtual void Action() const = 0;

    double Lanch() const {
        double resultTime = 0;
        for (size_t i = 0; i < ActionsNumber; ++i) {
            std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
            Action();
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            resultTime += static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
        }
        return resultTime;
    }

    std::tuple<double, double> Compute() const {
        std::vector<int> results(IterationNumber);
        std::generate_n(results.begin(), IterationNumber, [&]{ return Lanch(); });

        double mean = std::accumulate(results.begin(), results.end(), 0.0,
                                                         [&](double a, double b){ return a + b / IterationNumber; });

        double var = std::accumulate(results.begin(), results.end(), 0.0,
                                                        [&](double a, double b) { return a + (b - mean) * (b - mean) / IterationNumber; });

        return {mean, std::sqrt(var)};
    }

    virtual ~TProfiler() {}
};

class TSeqWriterProfiler : public TProfiler {
private:
    const size_t BlockSize;
    char* Buf;
public:

    TSeqWriterProfiler()
        : TProfiler()
        , BlockSize(1024 * 1024)
        , Buf(new char[BlockSize])
    {

        std::mt19937 gen{ std::random_device()() };
        std::uniform_int_distribution<> dis(0, 255);
        std::generate_n(Buf, BlockSize, [&]{ return dis(gen); });
        Buf[BlockSize - 1] = '\n';
    }

    void Action() const {
//        std::cout << "Action" << std::endl;
        // что если перенести открытие и закрытие в конструктор и дестркутор?
        int fd = open("file", O_DIRECT | O_SYNC);
        write(fd, Buf, BlockSize);
        close(fd);
    }

    ~TSeqWriterProfiler() {
        delete Buf;
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

    TSeqWriterProfiler profiler;
    double mean, var;

    std::tie(mean, var) = profiler.Compute();
    std::cout << mean << " " << var << std::endl;
    std::cout << 1. / mean  * 1000 * 1000 * 1024 << std::endl;

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
