cmake_minimum_required(VERSION 3.15)
project(msa_band_neural CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# pybind11 via pip: pip install pybind11
find_package(pybind11 REQUIRED)
find_package(OpenMP)
include(CheckCXXCompilerFlag)

# All source files
# Files #include each other in chain: full_nw → banded_nw → simd → four_russians → hirschberg → band_doubling
# band_doubling.cpp is the single compilation unit that pulls in the entire chain via #include.
# profile_dp.cpp and anchored_align.cpp use symbols from the chain but are compiled as separate units
# with their own includes guarded — they only provide additional helper functions.
set(ALIGNER_SOURCES
    aligner/band_doubling.cpp
)

pybind11_add_module(aligner ${ALIGNER_SOURCES})

# Compiler-specific flags
if(MSVC)
    target_compile_options(aligner PRIVATE /O2 /DNDEBUG)
    # AVX2 for MSVC
    check_cxx_compiler_flag("/arch:AVX2" COMPILER_HAS_AVX2)
    if(COMPILER_HAS_AVX2)
        target_compile_options(aligner PRIVATE /arch:AVX2)
        target_compile_definitions(aligner PRIVATE HAVE_AVX2)
        message(STATUS "AVX2 enabled (MSVC)")
    endif()
else()
    target_compile_options(aligner PRIVATE -O3 -march=native -DNDEBUG)
    # AVX2 for GCC/Clang
    check_cxx_compiler_flag("-mavx2" COMPILER_HAS_AVX2)
    if(COMPILER_HAS_AVX2)
        target_compile_options(aligner PRIVATE -mavx2)
        target_compile_definitions(aligner PRIVATE HAVE_AVX2)
        message(STATUS "AVX2 enabled")
    endif()
endif()

# OpenMP
if(OpenMP_CXX_FOUND)
    target_link_libraries(aligner PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(aligner PRIVATE HAVE_OPENMP)
    message(STATUS "OpenMP enabled")
endif()

# Install .pyd/.so next to Python code (project root)
set_target_properties(aligner PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
