Pedestrian Detector Version 1.0 10/12/2016

DESCRIPTION
------------
- Pedestrian detection is CUDA application which uses OpenCV 3.1.0 for detecting pedestrians on road fron given input.
- The input can be in the form of video, image or live stream

DIRECTORY STRUCTURE
--------------------
pedestrianDetection
---Demovideo.mp4: video for demo
---Input: input data
   ----data: contain the cascade.xml file
   ----neg:negative images 
   ----pos: positive images
   ----Test:testing images and demo video
   ----persons.info:information about positive samples
   ----bg.txt:information about negative samples
   ----persons.vec:contain samples created by opencv_createsamples
---Source: source files
   ----pedestrianDetector.cpp :contain source code
   ----tick_meter.hpp : contain functions for time calculation
---Makefile : 
   ----CMakeLists.txt : Cmake file

GUIDELINES FOR EXECUTION
-------------------------
1. For training (in input directory)
   > opencv_createsamples –info persons.info –num 1300 –w 24 –h 48 –vec persons.vec
   > opencv_traincascade –numStages 26 –data data –vec persons.vec   –bg bg.txt –numPos 1000 –numNeg 1000 –w 24 –h 48 –featureType LBP

2. Compile the source code(in MakeFile directory):
   > cmake .
   > make

3. For execution (in Team14 directory)
   > ./pedestrianDetector  --casade input/data/cascade.xml –video demovideo.mp4

