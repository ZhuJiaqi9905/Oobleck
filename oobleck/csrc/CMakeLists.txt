cmake_minimum_required(VERSION 3.8)

project(Oobleck)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_subdirectory(modules/cppcoro)
add_subdirectory(planning)