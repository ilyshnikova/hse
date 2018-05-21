#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <iterator>
#include <functional>
#include <algorithm>
#include <iomanip>
#include <sstream>

class ExecuteHandler {
private:
    std::vector<std::string> result;
    size_t index;
public:
    ExecuteHandler(const std::string& cmd)
        : result()
        , index(0)
    {
        std::cerr << cmd << "\n";
        FILE* pipe = popen(cmd.c_str(), "r");
        if (!pipe) {
            std::cerr << "Error while opening pipe\n";
            return;
        }
        char buffer[128];
        while(!feof(pipe)) {
            if(fgets(buffer, 128, pipe) != NULL) {
               result.push_back(static_cast<std::string>(buffer));
            }
        }
        pclose(pipe);
    }

    bool operator >> (std::string& string) {
        if (index == result.size()) {
            return 0;
        } else {
            string = result[index++];
            string.erase(
                string.begin(),
                std::find_if(
                    string.begin(),
                    string.end(),
                    std::not1(std::ptr_fun<int, int>(std::isspace))
                )
            );
            string.erase(
                std::find_if(
                    string.rbegin(),
                    string.rend(),
                    std::not1(std::ptr_fun<int, int>(std::isspace))
                ).base(),
                string.end()
            );
            std::cerr << string << std::endl;
            return 1;
        }
    }
};

std::string createCommand(const std::string& mode, const std::string& filtration_rate, const std::string& function_call) {
    return "./build/dbms/src/Server/clickhouse-client --stacktrace --time --multiquery --query=\"set enable_conditional_computation = " +
        mode +
        "; select count(*) from (select (number % " +
        filtration_rate +
        " == 0 and " +
        "(" + function_call + " == "+ function_call + ")" +
        ") from system.numbers limit 10000000)\" 2>&1";
}

double getTime(const std::string& mode, const std::string filtration_rate, const std::string& function_call) {
    std::string s_res;
    double res = 0;
    for (size_t i = 0; i < 5; ++i) {
        ExecuteHandler ex(createCommand(mode, filtration_rate, function_call));
        if (ex >> s_res && ex >> s_res && ex >> s_res) {
            s_res.erase(s_res.find_last_not_of(" \n\r\t")+1);
            try {
                res += std::stod(s_res);
            } catch (std::exception& e) {
                std::cout << mode << " " << function_call  << " " << filtration_rate << std::endl;
                std::cout << "faile" << std::endl;
            }
        } else {
            std::cout << mode << " " << function_call  << " " << filtration_rate << std::endl;
            throw std::runtime_error("query failed");
        }
    }
    return res / 5;
}

std::string getDiv(const double val) {
    if (val < 0.9) {
        return "\033[1;31m" + std::to_string(val) + "\033[0m";
    } else if (val > 1.1) {
        return "\033[1;32m" + std::to_string(val) + "\033[0m";
    }
    return std::to_string(val);
}

std::string getTexColor(const double val) {
    if (val < 0.9) {
        return "\\cellcolor{red!25} ";
    } else if (val > 1.1) {
        return "\\cellcolor{green!25} ";
    }
    return "";
}

std::string getTexDouble(const double val) {
    std::stringstream stream;
    stream << std::fixed << std::setprecision(2) << val;
    return stream.str();
}

void measureSpeed() {
    std::ofstream out("speed_res.tex");
    int rates_count;
    std::cin >> rates_count;
    std::vector<int> rates(rates_count);

    std::cout << "function";
    out << "\\begin{tabular}{l | *{" << rates_count << "}{c}}" << std::endl;
    out << "Function\t& ";
    for (size_t i = 0; i < rates_count; ++i) {
        std::cin >> rates[i];
        std::cout << "\t| rate(" << rates[i] << ")\t1\t0";
        out << "1/" << rates[i];
        if (i + 1 == rates_count) {
            out << "\\\\ \\hline" << std::endl;
            std::cout << std::endl;
        } else {
            out << "\t& ";
        }
    }

    int func_count;
    std::cin >> func_count;

    for (size_t i = 0; i < func_count; ++i) {
        std::string function_name;
        std::string function_call;
        std::cin >> function_name >> function_call;
        std::cout << function_name;
        out << function_name;
        if (function_name.size() < 8) {
            std::cout << "\t";
        }
        std::string first_line;
        for (int rate: rates) {
            double mode1 = getTime("1", std::to_string(rate), function_call);
            double mode0 = getTime("0", std::to_string(rate), function_call);
            double res = mode0 / mode1;
            std::cout << "\t| " << getDiv(res) << "\t"<<  mode1 << "\t" << mode0;
            first_line += "\t&" + getTexColor(res) + getTexDouble(res) + " (" + getTexDouble(mode0) + "/" + getTexDouble(mode1) + ")";
        }
        out << first_line << "\\\\ \\hline" << std::endl;
        std::cout << std::endl;
    }

    out << "\\end{tabular}" << std::endl;
    out.close();
}

int main() {
    measureSpeed();
    return 0;
}

