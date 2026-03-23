cmake_minimum_required(VERSION 3.16)
project(ml_kernel_project C)

set(CMAKE_C_STANDARD 99)

add_executable(test_kernel
    src/main.c
    src/naive/fc_int8.c
)

add_executable(test_fc
    src/tests/test_fc.c
    src/naive/fc_int8.c
)

add_executable(bench_fc
    src/tests/bench_fc.c
    src/naive/fc_int8.c
)

add_executable(test_fc_opt
    src/tests/test_fc_opt.c
    src/optimized/fc_int8_opt.c
)

add_executable(bench_fc_opt
    src/tests/bench_fc_opt.c
    src/optimized/fc_int8_opt.c
)

add_executable(test_dwconv
    src/tests/test_dwconv.c
    src/naive/dwconv_int8.c
)

add_executable(bench_dwconv
    src/tests/bench_dwconv.c
    src/naive/dwconv_int8.c
)

add_executable(test_dwconv_opt
    src/tests/test_dwconv_opt.c
    src/optimized/dwconv_int8_opt.c
)

add_executable(bench_dwconv_opt
    src/tests/bench_dwconv_opt.c
    src/optimized/dwconv_int8_opt.c
)

target_include_directories(test_dwconv_opt PRIVATE src)
target_include_directories(bench_dwconv_opt PRIVATE src)

target_include_directories(test_dwconv PRIVATE src)
target_include_directories(bench_dwconv PRIVATE src)
target_include_directories(test_kernel PRIVATE src)
target_include_directories(test_fc PRIVATE src)
target_include_directories(bench_fc PRIVATE src)
target_include_directories(test_fc_opt PRIVATE src)
target_include_directories(bench_fc_opt PRIVATE src)
