# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.7

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /opt/github/public/selfdrivingnd/term2/lessons/lesson5

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /opt/github/public/selfdrivingnd/term2/lessons/lesson5/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/lesson51.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/lesson51.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/lesson51.dir/flags.make

CMakeFiles/lesson51.dir/laser-measurement/main.cpp.o: CMakeFiles/lesson51.dir/flags.make
CMakeFiles/lesson51.dir/laser-measurement/main.cpp.o: ../laser-measurement/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/opt/github/public/selfdrivingnd/term2/lessons/lesson5/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/lesson51.dir/laser-measurement/main.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/lesson51.dir/laser-measurement/main.cpp.o -c /opt/github/public/selfdrivingnd/term2/lessons/lesson5/laser-measurement/main.cpp

CMakeFiles/lesson51.dir/laser-measurement/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/lesson51.dir/laser-measurement/main.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /opt/github/public/selfdrivingnd/term2/lessons/lesson5/laser-measurement/main.cpp > CMakeFiles/lesson51.dir/laser-measurement/main.cpp.i

CMakeFiles/lesson51.dir/laser-measurement/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/lesson51.dir/laser-measurement/main.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /opt/github/public/selfdrivingnd/term2/lessons/lesson5/laser-measurement/main.cpp -o CMakeFiles/lesson51.dir/laser-measurement/main.cpp.s

CMakeFiles/lesson51.dir/laser-measurement/main.cpp.o.requires:

.PHONY : CMakeFiles/lesson51.dir/laser-measurement/main.cpp.o.requires

CMakeFiles/lesson51.dir/laser-measurement/main.cpp.o.provides: CMakeFiles/lesson51.dir/laser-measurement/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/lesson51.dir/build.make CMakeFiles/lesson51.dir/laser-measurement/main.cpp.o.provides.build
.PHONY : CMakeFiles/lesson51.dir/laser-measurement/main.cpp.o.provides

CMakeFiles/lesson51.dir/laser-measurement/main.cpp.o.provides.build: CMakeFiles/lesson51.dir/laser-measurement/main.cpp.o


CMakeFiles/lesson51.dir/laser-measurement/KalmanFilter.cpp.o: CMakeFiles/lesson51.dir/flags.make
CMakeFiles/lesson51.dir/laser-measurement/KalmanFilter.cpp.o: ../laser-measurement/KalmanFilter.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/opt/github/public/selfdrivingnd/term2/lessons/lesson5/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/lesson51.dir/laser-measurement/KalmanFilter.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/lesson51.dir/laser-measurement/KalmanFilter.cpp.o -c /opt/github/public/selfdrivingnd/term2/lessons/lesson5/laser-measurement/KalmanFilter.cpp

CMakeFiles/lesson51.dir/laser-measurement/KalmanFilter.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/lesson51.dir/laser-measurement/KalmanFilter.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /opt/github/public/selfdrivingnd/term2/lessons/lesson5/laser-measurement/KalmanFilter.cpp > CMakeFiles/lesson51.dir/laser-measurement/KalmanFilter.cpp.i

CMakeFiles/lesson51.dir/laser-measurement/KalmanFilter.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/lesson51.dir/laser-measurement/KalmanFilter.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /opt/github/public/selfdrivingnd/term2/lessons/lesson5/laser-measurement/KalmanFilter.cpp -o CMakeFiles/lesson51.dir/laser-measurement/KalmanFilter.cpp.s

CMakeFiles/lesson51.dir/laser-measurement/KalmanFilter.cpp.o.requires:

.PHONY : CMakeFiles/lesson51.dir/laser-measurement/KalmanFilter.cpp.o.requires

CMakeFiles/lesson51.dir/laser-measurement/KalmanFilter.cpp.o.provides: CMakeFiles/lesson51.dir/laser-measurement/KalmanFilter.cpp.o.requires
	$(MAKE) -f CMakeFiles/lesson51.dir/build.make CMakeFiles/lesson51.dir/laser-measurement/KalmanFilter.cpp.o.provides.build
.PHONY : CMakeFiles/lesson51.dir/laser-measurement/KalmanFilter.cpp.o.provides

CMakeFiles/lesson51.dir/laser-measurement/KalmanFilter.cpp.o.provides.build: CMakeFiles/lesson51.dir/laser-measurement/KalmanFilter.cpp.o


CMakeFiles/lesson51.dir/laser-measurement/Tracking.cpp.o: CMakeFiles/lesson51.dir/flags.make
CMakeFiles/lesson51.dir/laser-measurement/Tracking.cpp.o: ../laser-measurement/Tracking.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/opt/github/public/selfdrivingnd/term2/lessons/lesson5/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/lesson51.dir/laser-measurement/Tracking.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/lesson51.dir/laser-measurement/Tracking.cpp.o -c /opt/github/public/selfdrivingnd/term2/lessons/lesson5/laser-measurement/Tracking.cpp

CMakeFiles/lesson51.dir/laser-measurement/Tracking.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/lesson51.dir/laser-measurement/Tracking.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /opt/github/public/selfdrivingnd/term2/lessons/lesson5/laser-measurement/Tracking.cpp > CMakeFiles/lesson51.dir/laser-measurement/Tracking.cpp.i

CMakeFiles/lesson51.dir/laser-measurement/Tracking.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/lesson51.dir/laser-measurement/Tracking.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /opt/github/public/selfdrivingnd/term2/lessons/lesson5/laser-measurement/Tracking.cpp -o CMakeFiles/lesson51.dir/laser-measurement/Tracking.cpp.s

CMakeFiles/lesson51.dir/laser-measurement/Tracking.cpp.o.requires:

.PHONY : CMakeFiles/lesson51.dir/laser-measurement/Tracking.cpp.o.requires

CMakeFiles/lesson51.dir/laser-measurement/Tracking.cpp.o.provides: CMakeFiles/lesson51.dir/laser-measurement/Tracking.cpp.o.requires
	$(MAKE) -f CMakeFiles/lesson51.dir/build.make CMakeFiles/lesson51.dir/laser-measurement/Tracking.cpp.o.provides.build
.PHONY : CMakeFiles/lesson51.dir/laser-measurement/Tracking.cpp.o.provides

CMakeFiles/lesson51.dir/laser-measurement/Tracking.cpp.o.provides.build: CMakeFiles/lesson51.dir/laser-measurement/Tracking.cpp.o


# Object files for target lesson51
lesson51_OBJECTS = \
"CMakeFiles/lesson51.dir/laser-measurement/main.cpp.o" \
"CMakeFiles/lesson51.dir/laser-measurement/KalmanFilter.cpp.o" \
"CMakeFiles/lesson51.dir/laser-measurement/Tracking.cpp.o"

# External object files for target lesson51
lesson51_EXTERNAL_OBJECTS =

lesson51: CMakeFiles/lesson51.dir/laser-measurement/main.cpp.o
lesson51: CMakeFiles/lesson51.dir/laser-measurement/KalmanFilter.cpp.o
lesson51: CMakeFiles/lesson51.dir/laser-measurement/Tracking.cpp.o
lesson51: CMakeFiles/lesson51.dir/build.make
lesson51: CMakeFiles/lesson51.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/opt/github/public/selfdrivingnd/term2/lessons/lesson5/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable lesson51"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/lesson51.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/lesson51.dir/build: lesson51

.PHONY : CMakeFiles/lesson51.dir/build

CMakeFiles/lesson51.dir/requires: CMakeFiles/lesson51.dir/laser-measurement/main.cpp.o.requires
CMakeFiles/lesson51.dir/requires: CMakeFiles/lesson51.dir/laser-measurement/KalmanFilter.cpp.o.requires
CMakeFiles/lesson51.dir/requires: CMakeFiles/lesson51.dir/laser-measurement/Tracking.cpp.o.requires

.PHONY : CMakeFiles/lesson51.dir/requires

CMakeFiles/lesson51.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/lesson51.dir/cmake_clean.cmake
.PHONY : CMakeFiles/lesson51.dir/clean

CMakeFiles/lesson51.dir/depend:
	cd /opt/github/public/selfdrivingnd/term2/lessons/lesson5/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /opt/github/public/selfdrivingnd/term2/lessons/lesson5 /opt/github/public/selfdrivingnd/term2/lessons/lesson5 /opt/github/public/selfdrivingnd/term2/lessons/lesson5/cmake-build-debug /opt/github/public/selfdrivingnd/term2/lessons/lesson5/cmake-build-debug /opt/github/public/selfdrivingnd/term2/lessons/lesson5/cmake-build-debug/CMakeFiles/lesson51.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/lesson51.dir/depend

