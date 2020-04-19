# Set a default build type if none was specified A build type indicates a set of compile-time decisions used while
# compiling your code. For example, whether or not to optimize or to include debug information. A build type or build
# mode or configuration is popular in Visual Studio, where C++ projects typically have at least two types: Release and
# Debug.

# When no build type is provided, I found the compilation string to be: g++ /your/code.cpp For Release build type I
# found the compilation string to be: g++ -O3 -DNDEBUG /your/code.cpp. Note that assert will be disabled with this build
# type. For Debug build type, I found the compilation string to be: g++ -g /your/code.cpp

# CMake has build types and follows similar rules. For compiling C++, there are at least 4 build types in CMake:
# Release, Debug, MinSizeRel and RelWithDebInfo.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'RelWithDebInfo' as none was specified.")
  set(CMAKE_BUILD_TYPE
      RelWithDebInfo
      CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui, ccmake
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Ccache (or “ccache”) is a compiler cache. It speeds up recompilation by caching previous compilations and detecting
# when the same compilation is being done again.
find_program(CCACHE ccache)
if(CCACHE)
  message("using ccache")
  set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE})
else()
  message("ccache not found cannot use")
endif()

# Generate compile_commands.json to make it easier to work with clang based tools
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Interprocedural optimization (IPO) is a collection of compiler techniques used in computer programming to improve
# performance in programs containing many frequently used functions of small or medium length. IPO differs from other
# compiler optimization because it analyzes the entire program; other optimizations look at only a single function, or
# even a single block of code.
option(ENABLE_IPO "Enable Iterprocedural Optimization, aka Link Time Optimization (LTO)" OFF)

if(ENABLE_IPO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT result OUTPUT output)
  if(result)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
  else()
    message(SEND_ERROR "IPO is not supported: ${output}")
  endif()
endif()
