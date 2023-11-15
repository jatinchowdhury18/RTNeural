#pragma once

#include <fstream>
#include <sstream>
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

template <typename T, int N>
std::vector<std::vector<T>> loadFile2d(std::ifstream& stream)
{
    std::vector<std::vector<T>> output;
    
    std::string line;
    while(std::getline(stream, line)) {
        std::vector<T> row;
        std::stringstream lineStream(line);
        std::string cell;

        while(std::getline(lineStream, cell, ',')) {
            row.push_back(static_cast<T>(std::stod(cell)));
        }

        output.push_back(row);
    }

    return output;
}

} // namespace load_csv
