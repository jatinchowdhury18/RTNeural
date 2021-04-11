#pragma once

#include <fstream>
#include <string>
#include <vector>

namespace load_csv {

template <typename T>
std::vector<T> loadFile(std::ifstream& stream)
{
    std::vector<T> vec;

    std::string line;
    if(stream.is_open()) {
        while(std::getline(stream, line))
            vec.push_back(static_cast<T>(std::stod(line)));

        stream.close();
    }

    return vec;
}

} // namespace load_csv
