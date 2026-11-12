cmake_minimum_required(VERSION 3.16.3)

project(arx5-sdk)

set(CMAKE_BUILD_TYPE Debug)
option(ARX5_BUILD_CPP_TESTS "Build standalone C++ test executables" OFF)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # global PIC

find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(orocos_kdl REQUIRED)
find_package(kdl_parser REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(Boost REQUIRED COMPONENTS container)
find_package(tinyxml2 REQUIRED)

if(TARGET kdl_parser::kdl_parser)
    set(KDL_PARSER_LIB kdl_parser::kdl_parser)
else()
    set(KDL_PARSER_LIB kdl_parser)
endif()

find_library(SOEM_LIBRARY
    NAMES soem
    HINTS
        $ENV{CONDA_PREFIX}/lib
    REQUIRED
)

add_compile_definitions(SDK_ROOT="${CMAKE_BINARY_DIR}/..")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/SelectSdkLibDir.cmake)
arx5_resolve_lib_dir(${CMAKE_CURRENT_SOURCE_DIR} LIB_DIR)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(
    $ENV{CONDA_PREFIX}/include/
    $ENV{CONDA_PREFIX}/include/urdfdom_headers
)

add_library(ArxJointController SHARED
    src/app/joint_controller.cpp
    src/app/controller_base.cpp
    src/app/friction_config.cpp
    src/utils.cpp
)
target_link_libraries(ArxJointController
    ${LIB_DIR}/libhardware.so
    ${LIB_DIR}/libsolver.so
    Eigen3::Eigen
    Threads::Threads
    spdlog::spdlog
    tinyxml2::tinyxml2
    ${KDL_PARSER_LIB}
    orocos-kdl
    ${SOEM_LIBRARY}
)
add_library(ArxCartesianController SHARED
    src/app/cartesian_controller.cpp
    src/app/cartesian_ik.cpp
    src/app/controller_base.cpp
    src/app/cartesian_pd.cpp
    src/app/friction_config.cpp
    src/utils.cpp
)
target_link_libraries(ArxCartesianController
    ${LIB_DIR}/libhardware.so
    ${LIB_DIR}/libsolver.so
    Eigen3::Eigen
    Threads::Threads
    spdlog::spdlog
    tinyxml2::tinyxml2
    ${KDL_PARSER_LIB}
    orocos-kdl
    ${SOEM_LIBRARY}
)

# Hack for py310 (conda environment is slightly different from other python versions)
target_include_directories(ArxJointController PUBLIC $ENV{CONDA_PREFIX}/include/kdl_parser $ENV{CONDA_PREFIX}/include/urdfdom_headers)
target_include_directories(ArxCartesianController PUBLIC $ENV{CONDA_PREFIX}/include/kdl_parser $ENV{CONDA_PREFIX}/include/urdfdom_headers)


if(ARX5_BUILD_CPP_TESTS)
    add_executable(test_joint_controller examples/test_joint_controller.cpp)
    target_link_libraries(test_joint_controller
        ${LIB_DIR}/libhardware.so
        ${LIB_DIR}/libsolver.so
        ArxJointController
        Eigen3::Eigen
        Threads::Threads
        spdlog::spdlog
        ${KDL_PARSER_LIB}
        orocos-kdl
        ${SOEM_LIBRARY}
    )

    add_executable(test_cartesian_controller examples/test_cartesian_controller.cpp)
    target_link_libraries(test_cartesian_controller
        ${LIB_DIR}/libhardware.so
        ${LIB_DIR}/libsolver.so
        ArxCartesianController
        spdlog::spdlog
        Eigen3::Eigen
        Threads::Threads
        ${KDL_PARSER_LIB}
        orocos-kdl
        ${SOEM_LIBRARY}
    )

    add_executable(test_gripper_friction_compensation examples/test_gripper_friction_compensation.cpp)
    target_link_libraries(test_gripper_friction_compensation
        ${LIB_DIR}/libhardware.so
        ${LIB_DIR}/libsolver.so
        ArxJointController
        Eigen3::Eigen
        Threads::Threads
        spdlog::spdlog
        ${KDL_PARSER_LIB}
        orocos-kdl
        ${SOEM_LIBRARY}
    )

    enable_testing()
    add_executable(test_cartesian_pd
        tests/test_cartesian_pd.cpp
        src/app/cartesian_pd.cpp
        src/app/cartesian_ik.cpp
        src/app/friction_config.cpp
        src/utils.cpp
    )
    target_include_directories(test_cartesian_pd PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/app)
    target_link_libraries(test_cartesian_pd
        Eigen3::Eigen
        ${KDL_PARSER_LIB}
        orocos-kdl
        tinyxml2::tinyxml2
        ${CMAKE_DL_LIBS}
    )
    add_test(NAME cartesian_pd_unit_tests COMMAND test_cartesian_pd)
    set_tests_properties(cartesian_pd_unit_tests PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        ENVIRONMENT "AMENT_PREFIX_PATH=$ENV{CONDA_PREFIX}"
    )
endif()

add_subdirectory(python)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}
)

install(FILES ${LIB_DIR}/libhardware.so ${LIB_DIR}/libsolver.so
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME}
)

install(TARGETS ArxJointController ArxCartesianController
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME}
)
