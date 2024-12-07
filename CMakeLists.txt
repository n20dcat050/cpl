cmake_minimum_required(VERSION 3.10)
project(MachineLearningWithCpp)

set(CMAKE_CXX_STANDARD 14)

# Tìm thư viện Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Thêm thư viện RapidCSV
include_directories(${CMAKE_SOURCE_DIR}/rapidcsv/src)

# Thêm mã nguồn
add_executable(main src/main.cpp)
