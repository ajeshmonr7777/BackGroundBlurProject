#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>


using namespace std;
using namespace cv;
using namespace dnn;

vector<string> trackerTypes = { "BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "GOTURN", "MOSSE", "CSRT" };

// create tracker by name
Ptr<Tracker> createTrackerByName(string trackerType)
{
    Ptr<Tracker> tracker;
    if (trackerType == trackerTypes[0])
        tracker = TrackerBoosting::create();
    else if (trackerType == trackerTypes[1])
        tracker = TrackerMIL::create();
    else if (trackerType == trackerTypes[2])
        tracker = TrackerKCF::create();
    else if (trackerType == trackerTypes[3])
        tracker = TrackerTLD::create();
    else if (trackerType == trackerTypes[4])
        tracker = TrackerMedianFlow::create();
    else if (trackerType == trackerTypes[5])
        tracker = TrackerGOTURN::create();
    else if (trackerType == trackerTypes[6])
        tracker = TrackerMOSSE::create();
    else if (trackerType == trackerTypes[7])
        tracker = TrackerCSRT::create();
    else {
        cout << "Incorrect tracker name" << endl;
        cout << "Available trackers are: " << endl;
        for (vector<string>::iterator it = trackerTypes.begin(); it != trackerTypes.end(); ++it)
            std::cout << " " << *it << endl;
    }
    return tracker;
}

// Fill the vector with random colors
void getRandomColors(vector<Scalar>& colors, int numColors)
{
    RNG rng(1);
    for (int i = 0; i < numColors; i++)
        colors.push_back(Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)));
}

vector<Rect> get_bboxes(Mat output, vector<Rect> bboxes, Mat frame)
{


    Mat detectionMat(output.size[2], output.size[3], CV_32F, output.ptr<float>());

    for (int i = 0; i < detectionMat.rows; i++) {
        int class_id = detectionMat.at<float>(i, 1);
        float confidence = detectionMat.at<float>(i, 2);

        // Check if the detection is of good quality

        if (confidence > 0.2 && class_id == 1) {
            int box_x = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
            int box_y = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
            int box_width = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols - box_x);
            int box_height = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows - box_y);
            //rectangle(frame, Point(box_x, box_y), Point(box_x + box_width, box_y + box_height), Scalar(255, 0, 255), 2);
            //putText(image, class_names[class_id - 1].c_str(), Point(box_x, box_y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 0), 1);
            Rect box = Rect(box_x, box_y, box_width, box_height);
            bboxes.push_back(box);
            cout << bboxes[i] << endl;
        }

    }
    return bboxes;
}

vector<Rect> model_inference(Mat frame, vector<Rect> bboxes, Net model) {


    bboxes = {};
    //create blob from image
    Mat blob = blobFromImage(frame, 1.0, Size(300, 300), Scalar(127.5, 127.5, 127.5),
        true, false);
    model.setInput(blob);
    //forward pass through the model to carry out the detection
    Mat output = model.forward();


    //getting vector of bounding boxes
    bboxes = get_bboxes(output, bboxes, frame);
    return bboxes;
}

Net load_models() {

    std::vector<std::string> class_names;
    ifstream ifs(string("resources/object_detection_classes_coco.txt").c_str());
    string line;
    while (getline(ifs, line))
    {
        class_names.push_back(line);
    }

    // load the neural network model
    auto model = readNet("resources/frozen_inference_graph.pb",
        "resources/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt",
        "TensorFlow");

    return model;
}

int main(int argc, char* argv[])
{
    cout << "Default tracking algoritm is CSRT" << endl;
    cout << "Available tracking algorithms are:" << endl;
    for (vector<string>::iterator it = trackerTypes.begin(); it != trackerTypes.end(); ++it)
        std::cout << " " << *it << endl;

    // Set tracker type. Change this to try different trackers.
    string trackerType = "MEDIANFLOW";

    // create a video capture object to read videos
    cv::VideoCapture cap(0);
    Mat frame;

    cap >> frame;

    // quit if unabke to read video file
    if (!cap.isOpened())
    {
        cout << "Error opening video file " << endl;
        return -1;
    }

    Net model = load_models();

    Mat BlurImg;

    int frame_number = 0;

    vector <Rect> bboxes;

    //create multitracker
    Ptr<MultiTracker> multiTracker;

    float sum = 0.0;

    while (cap.isOpened())
    {

        // Start timer
        double timer = (double)getTickCount();

        Mat board(frame.size().height, frame.size().width, CV_8U, Scalar(0, 0, 0));

        GaussianBlur(frame, BlurImg, Size(23, 23), 5, 5);

        // get frame from the video
        cap >> frame;

        // stop the program if reached end of video
        if (frame.empty()) break;

        if (frame_number % 70 == 0) {


            bboxes = model_inference(frame, bboxes, model);

            // quit if there are no objects to track
            if (bboxes.size() < 1) {
                return 0;
            }
            vector<Scalar> colors;
            getRandomColors(colors, bboxes.size());

            // Create multitracker
            MultiTracker* var = new MultiTracker;

            //multiTracker = MultiTracker::create();

            multiTracker = var->create();

            // initialize multitracker
            for (int i = 0; i < bboxes.size(); i++)
                multiTracker->add(createTrackerByName(trackerType), frame, Rect2d(bboxes[i]));

            delete var;
        }

        else {

            //update the tracking result with new frame
            multiTracker->update(frame);

            vector<Scalar> colors;
            getRandomColors(colors, bboxes.size());

            // draw tracked objects
            for (unsigned i = 0; i < multiTracker->getObjects().size(); i++)
            {
                rectangle(frame, multiTracker->getObjects()[i], colors[i], 1, 1);
                rectangle(board, multiTracker->getObjects()[i], (255, 255, 255), -1, 1);
            }

        }

        frame.copyTo(BlurImg, board);

        // Calculate Frames per second (FPS)
        float fps = getTickFrequency() / ((double)getTickCount() - timer);

        sum = sum + fps;
        float average_fps = sum / (frame_number + 1);


        if (frame_number % 300 == 0 && frame_number != 0) {
            sum = 0;
            frame_number = 0;
        }

        string label = format("FPS : %0.0f ", average_fps); // Display FPS on frame


        putText(BlurImg, label, Point(50, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255));

        //imshow("MultiTracker", frame);
        //imshow("Board", board);
        imshow("Blured", BlurImg);

        // quit on x button
        if (waitKey(1) == 27) break;
        frame_number++;
    }

}