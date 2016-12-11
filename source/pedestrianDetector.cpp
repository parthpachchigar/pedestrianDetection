// WARNING: this sample is under construction! Use it on your own risk.
#if defined _MSC_VER && _MSC_VER >= 1400
#pragma warning(disable : 4100)
#endif


#include <iostream>
#include <iomanip>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/cudaobjdetect.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"

#include "tick_meter.hpp"

using namespace std;
using namespace cv;
using namespace cv::cuda;


static void help()
{
    cout << "Usage: ./cascadeclassifier_gpu \n\t--cascade <cascade_file>\n\t(<image>|--video <video>|--camera <camera_id>)\n"
            "Using OpenCV version " << CV_VERSION << endl << endl;
}


static void convertAndResize(const Mat& src, Mat& gray, Mat& resized, double scale)
{
    if (src.channels() == 3)
    {
        cv::cvtColor( src, gray, COLOR_BGR2GRAY );
    }
    else
    {
        gray = src;
    }

    Size sz(cvRound(gray.cols * scale), cvRound(gray.rows * scale));

    if (scale != 1)
    {
        cv::resize(gray, resized, sz);
    }
    else
    {
        resized = gray;
    }
}

static void convertAndResize(const GpuMat& src, GpuMat& gray, GpuMat& resized, double scale)
{
    if (src.channels() == 3)
    {
        cv::cuda::cvtColor( src, gray, COLOR_BGR2GRAY );
    }
    else
    {
        gray = src;
    }

    Size sz(cvRound(gray.cols * scale), cvRound(gray.rows * scale));

    if (scale != 1)
    {
        cv::cuda::resize(gray, resized, sz);
    }
    else
    {
        resized = gray;
    }
}







int main(int argc, const char *argv[])
{
    if (argc == 1)
    {
        help();
        return -1;
    }

    if (getCudaEnabledDeviceCount() == 0)
    {
        return cerr << "No GPU found or the library is compiled without CUDA support" << endl, -1;
    }

    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

    string cascadeName;
    string inputName;
    bool isInputImage = false;
    bool isInputVideo = false;
    bool isInputCamera = false;

    for (int i = 1; i < argc; ++i)
    {
        if (string(argv[i]) == "--cascade")
            cascadeName = argv[++i];
        else if (string(argv[i]) == "--video")
        {
            inputName = argv[++i];
            isInputVideo = true;
        }
        else if (string(argv[i]) == "--camera")
        {
            inputName = argv[++i];
            isInputCamera = true;
        }
        else if (string(argv[i]) == "--help")
        {
            help();
            return -1;
        }
        else if (!isInputImage)
        {
            inputName = argv[i];
            isInputImage = true;
        }
        else
        {
            cout << "Unknown key: " << argv[i] << endl;
	    help();
            return -1;
        }
    }

    Ptr<cuda::CascadeClassifier> cascade_gpu = cuda::CascadeClassifier::create(cascadeName);

    cv::CascadeClassifier cascade_cpu;
    if (!cascade_cpu.load(cascadeName))
    {
        return cerr << "ERROR: Could not load cascade classifier \"" << cascadeName << "\"" << endl, help(), -1;
    }

    VideoCapture capture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)I420, framrate=(fraction)24/1 ! nvvidconv flip-method=2 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR !appsink");
    Mat image;

    if (isInputImage)
    {
        image = imread(inputName);
        CV_Assert(!image.empty());
    }
    else if (isInputVideo)
    {
        capture.open(inputName);
        CV_Assert(capture.isOpened());
    }
    else
    {
        //capture.open(atoi(inputName.c_str()));
        CV_Assert(capture.isOpened());
    }

    namedWindow("Pedestrian Detection Result", 1);

    Mat frame, frame_cpu, gray_cpu, resized_cpu, frameDisp;
    vector<Rect> persons;

    GpuMat frame_gpu, gray_gpu, resized_gpu, personsBuf_gpu;

    /* parameters */
    bool useGPU = true;
    double scaleFactor = 1.0;
    bool findLargestObject = true;
    bool filterRects = true;
    bool helpScreen = false;

    for (;;)
    {
        if (isInputCamera || isInputVideo)
        {
            capture >> frame;
            if (frame.empty())
            {
                break;
            }
        }

        (image.empty() ? frame : image).copyTo(frame_cpu);
        frame_gpu.upload(image.empty() ? frame : image);

        convertAndResize(frame_gpu, gray_gpu, resized_gpu, scaleFactor);
        convertAndResize(frame_cpu, gray_cpu, resized_cpu, scaleFactor);

        TickMeter tm;
        tm.start();
	Size minSize;
        if (useGPU)
        {
            cascade_gpu->setFindLargestObject(findLargestObject);
            cascade_gpu->setScaleFactor(1.2);
            cascade_gpu->setMinNeighbors((filterRects || findLargestObject) ? 4 : 0);

            cascade_gpu->detectMultiScale(resized_gpu, personsBuf_gpu);
            cascade_gpu->convert(personsBuf_gpu, persons);
        }
        else
        {
            minSize = cascade_gpu->getClassifierSize();
            cascade_cpu.detectMultiScale(resized_cpu, persons, 1.1,2,0,minSize);/*
                                         (filterRects || findLargestObject) ? 4 : 0,
                                         (findLargestObject ? CASCADE_FIND_BIGGEST_OBJECT : 0)
                                            | CASCADE_SCALE_IMAGE,
                                         minSize);
	    */
	    
        }

        for (size_t i = 0; i < persons.size(); ++i)
        {
           Mat personROI = resized_cpu(persons[i]);        
	   std::vector<Rect> person_focused;        
	   cascade_cpu.detectMultiScale( personROI, person_focused, 1.1, 2, 0, minSize);        
	   if(person_focused.size() >0)        
	   {            
		rectangle(resized_cpu, persons[i], Scalar(255));        
	   }    
        }

        tm.stop();
        double detectionTime = tm.getTimeMilli();
        double fps = 1000 / detectionTime;

        //print detections to console
        cout << setfill(' ') << setprecision(2);
        cout << setw(6) << fixed << fps << " FPS, " << persons.size() << " det";
        if ((filterRects || findLargestObject) && !persons.empty())
        {
            for (size_t i = 0; i < persons.size(); ++i)
            {
                cout << ", [" << setw(4) << persons[i].x
                     << ", " << setw(4) << persons[i].y
                     << ", " << setw(4) << persons[i].width
                     << ", " << setw(4) << persons[i].height << "]";
            }
        }
        cout << endl;

        cv::cvtColor(resized_cpu, frameDisp, COLOR_GRAY2BGR);
       imshow("Pedestrian Detection Result", frameDisp);

        char key = (char)waitKey(5);
        if (key == 27)
        {
            break;
        }

        switch (key)
        {
        case ' ':
            useGPU = !useGPU;
            break;
        case 'm':
        case 'M':
            findLargestObject = !findLargestObject;
            break;
        case 'f':
        case 'F':
            filterRects = !filterRects;
            break;
        case '1':
            scaleFactor *= 1.05;
            break;
        case 'q':
        case 'Q':
            scaleFactor /= 1.05;
            break;
       
        }
    }

    return 0;
}
