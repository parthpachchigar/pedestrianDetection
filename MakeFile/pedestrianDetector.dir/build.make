# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ubuntu/Team14/MakeFile

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ubuntu/Team14/MakeFile

# Include any dependencies generated for this target.
include CMakeFiles/../pedestrianDetector.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/../pedestrianDetector.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/../pedestrianDetector.dir/flags.make

CMakeFiles/../pedestrianDetector.dir/home/ubuntu/Team14/source/pedestrianDetector.cpp.o: CMakeFiles/../pedestrianDetector.dir/flags.make
CMakeFiles/../pedestrianDetector.dir/home/ubuntu/Team14/source/pedestrianDetector.cpp.o: /home/ubuntu/Team14/source/pedestrianDetector.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/ubuntu/Team14/MakeFile/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/../pedestrianDetector.dir/home/ubuntu/Team14/source/pedestrianDetector.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/../pedestrianDetector.dir/home/ubuntu/Team14/source/pedestrianDetector.cpp.o -c /home/ubuntu/Team14/source/pedestrianDetector.cpp

CMakeFiles/../pedestrianDetector.dir/home/ubuntu/Team14/source/pedestrianDetector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/../pedestrianDetector.dir/home/ubuntu/Team14/source/pedestrianDetector.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/ubuntu/Team14/source/pedestrianDetector.cpp > CMakeFiles/../pedestrianDetector.dir/home/ubuntu/Team14/source/pedestrianDetector.cpp.i

CMakeFiles/../pedestrianDetector.dir/home/ubuntu/Team14/source/pedestrianDetector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/../pedestrianDetector.dir/home/ubuntu/Team14/source/pedestrianDetector.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/ubuntu/Team14/source/pedestrianDetector.cpp -o CMakeFiles/../pedestrianDetector.dir/home/ubuntu/Team14/source/pedestrianDetector.cpp.s

CMakeFiles/../pedestrianDetector.dir/home/ubuntu/Team14/source/pedestrianDetector.cpp.o.requires:
.PHONY : CMakeFiles/../pedestrianDetector.dir/home/ubuntu/Team14/source/pedestrianDetector.cpp.o.requires

CMakeFiles/../pedestrianDetector.dir/home/ubuntu/Team14/source/pedestrianDetector.cpp.o.provides: CMakeFiles/../pedestrianDetector.dir/home/ubuntu/Team14/source/pedestrianDetector.cpp.o.requires
	$(MAKE) -f CMakeFiles/../pedestrianDetector.dir/build.make CMakeFiles/../pedestrianDetector.dir/home/ubuntu/Team14/source/pedestrianDetector.cpp.o.provides.build
.PHONY : CMakeFiles/../pedestrianDetector.dir/home/ubuntu/Team14/source/pedestrianDetector.cpp.o.provides

CMakeFiles/../pedestrianDetector.dir/home/ubuntu/Team14/source/pedestrianDetector.cpp.o.provides.build: CMakeFiles/../pedestrianDetector.dir/home/ubuntu/Team14/source/pedestrianDetector.cpp.o

# Object files for target ../pedestrianDetector
__/pedestrianDetector_OBJECTS = \
"CMakeFiles/../pedestrianDetector.dir/home/ubuntu/Team14/source/pedestrianDetector.cpp.o"

# External object files for target ../pedestrianDetector
__/pedestrianDetector_EXTERNAL_OBJECTS =

../pedestrianDetector: CMakeFiles/../pedestrianDetector.dir/home/ubuntu/Team14/source/pedestrianDetector.cpp.o
../pedestrianDetector: CMakeFiles/../pedestrianDetector.dir/build.make
../pedestrianDetector: /home/ubuntu/opencv-3.1.0/lib/libopencv_videostab.so.3.1.0
../pedestrianDetector: /home/ubuntu/opencv-3.1.0/lib/libopencv_videoio.so.3.1.0
../pedestrianDetector: /home/ubuntu/opencv-3.1.0/lib/libopencv_video.so.3.1.0
../pedestrianDetector: /home/ubuntu/opencv-3.1.0/lib/libopencv_superres.so.3.1.0
../pedestrianDetector: /home/ubuntu/opencv-3.1.0/lib/libopencv_stitching.so.3.1.0
../pedestrianDetector: /home/ubuntu/opencv-3.1.0/lib/libopencv_shape.so.3.1.0
../pedestrianDetector: /home/ubuntu/opencv-3.1.0/lib/libopencv_photo.so.3.1.0
../pedestrianDetector: /home/ubuntu/opencv-3.1.0/lib/libopencv_objdetect.so.3.1.0
../pedestrianDetector: /home/ubuntu/opencv-3.1.0/lib/libopencv_ml.so.3.1.0
../pedestrianDetector: /home/ubuntu/opencv-3.1.0/lib/libopencv_imgproc.so.3.1.0
../pedestrianDetector: /home/ubuntu/opencv-3.1.0/lib/libopencv_imgcodecs.so.3.1.0
../pedestrianDetector: /home/ubuntu/opencv-3.1.0/lib/libopencv_highgui.so.3.1.0
../pedestrianDetector: /home/ubuntu/opencv-3.1.0/lib/libopencv_flann.so.3.1.0
../pedestrianDetector: /home/ubuntu/opencv-3.1.0/lib/libopencv_features2d.so.3.1.0
../pedestrianDetector: /home/ubuntu/opencv-3.1.0/lib/libopencv_cudev.so.3.1.0
../pedestrianDetector: /home/ubuntu/opencv-3.1.0/lib/libopencv_cudawarping.so.3.1.0
../pedestrianDetector: /home/ubuntu/opencv-3.1.0/lib/libopencv_cudastereo.so.3.1.0
../pedestrianDetector: /home/ubuntu/opencv-3.1.0/lib/libopencv_cudaoptflow.so.3.1.0
../pedestrianDetector: /home/ubuntu/opencv-3.1.0/lib/libopencv_cudaobjdetect.so.3.1.0
../pedestrianDetector: /home/ubuntu/opencv-3.1.0/lib/libopencv_cudalegacy.so.3.1.0
../pedestrianDetector: /home/ubuntu/opencv-3.1.0/lib/libopencv_cudaimgproc.so.3.1.0
../pedestrianDetector: /home/ubuntu/opencv-3.1.0/lib/libopencv_cudafilters.so.3.1.0
../pedestrianDetector: /home/ubuntu/opencv-3.1.0/lib/libopencv_cudafeatures2d.so.3.1.0
../pedestrianDetector: /home/ubuntu/opencv-3.1.0/lib/libopencv_cudacodec.so.3.1.0
../pedestrianDetector: /home/ubuntu/opencv-3.1.0/lib/libopencv_cudabgsegm.so.3.1.0
../pedestrianDetector: /home/ubuntu/opencv-3.1.0/lib/libopencv_cudaarithm.so.3.1.0
../pedestrianDetector: /home/ubuntu/opencv-3.1.0/lib/libopencv_core.so.3.1.0
../pedestrianDetector: /home/ubuntu/opencv-3.1.0/lib/libopencv_calib3d.so.3.1.0
../pedestrianDetector: /home/ubuntu/opencv-3.1.0/lib/libopencv_cudawarping.so.3.1.0
../pedestrianDetector: /home/ubuntu/opencv-3.1.0/lib/libopencv_objdetect.so.3.1.0
../pedestrianDetector: /home/ubuntu/opencv-3.1.0/lib/libopencv_cudafilters.so.3.1.0
../pedestrianDetector: /home/ubuntu/opencv-3.1.0/lib/libopencv_cudaarithm.so.3.1.0
../pedestrianDetector: /home/ubuntu/opencv-3.1.0/lib/libopencv_features2d.so.3.1.0
../pedestrianDetector: /home/ubuntu/opencv-3.1.0/lib/libopencv_ml.so.3.1.0
../pedestrianDetector: /home/ubuntu/opencv-3.1.0/lib/libopencv_highgui.so.3.1.0
../pedestrianDetector: /home/ubuntu/opencv-3.1.0/lib/libopencv_videoio.so.3.1.0
../pedestrianDetector: /home/ubuntu/opencv-3.1.0/lib/libopencv_imgcodecs.so.3.1.0
../pedestrianDetector: /home/ubuntu/opencv-3.1.0/lib/libopencv_flann.so.3.1.0
../pedestrianDetector: /home/ubuntu/opencv-3.1.0/lib/libopencv_video.so.3.1.0
../pedestrianDetector: /home/ubuntu/opencv-3.1.0/lib/libopencv_imgproc.so.3.1.0
../pedestrianDetector: /home/ubuntu/opencv-3.1.0/lib/libopencv_core.so.3.1.0
../pedestrianDetector: /home/ubuntu/opencv-3.1.0/lib/libopencv_cudev.so.3.1.0
../pedestrianDetector: CMakeFiles/../pedestrianDetector.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable ../pedestrianDetector"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/../pedestrianDetector.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/../pedestrianDetector.dir/build: ../pedestrianDetector
.PHONY : CMakeFiles/../pedestrianDetector.dir/build

CMakeFiles/../pedestrianDetector.dir/requires: CMakeFiles/../pedestrianDetector.dir/home/ubuntu/Team14/source/pedestrianDetector.cpp.o.requires
.PHONY : CMakeFiles/../pedestrianDetector.dir/requires

CMakeFiles/../pedestrianDetector.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/../pedestrianDetector.dir/cmake_clean.cmake
.PHONY : CMakeFiles/../pedestrianDetector.dir/clean

CMakeFiles/../pedestrianDetector.dir/depend:
	cd /home/ubuntu/Team14/MakeFile && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ubuntu/Team14/MakeFile /home/ubuntu/Team14/MakeFile /home/ubuntu/Team14/MakeFile /home/ubuntu/Team14/MakeFile /home/ubuntu/Team14/MakeFile/pedestrianDetector.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/../pedestrianDetector.dir/depend
