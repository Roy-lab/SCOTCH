# Minimum CMake version
cmake_minimum_required(VERSION 3.15)

# Project name and settings
project(PyEnrichAnalyzer LANGUAGES CXX)

# Set C++ standard (Pybind11 works well for C++14, 17, or 20)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the Pybind11 package

file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.C")


# Add your source files
add_library(${PROJECT_NAME} MODULE ${SOURCES}
        ValueItem.cpp
        ValueItem.h
)


set(pybind11_DIR "/opt/anaconda3/envs/PyEnrichAnalyzer/share/cmake/pybind11")
find_package(pybind11 REQUIRED)
set(PYBIND11_FINDPYTHON ON)


# Link Pybind11 to your target
target_link_libraries(${PROJECT_NAME} PRIVATE pybind11::module)

# Optional: If using Conda and Pybind11 isn't found automatically
# You can uncomment and add the include directory manually if needed:

