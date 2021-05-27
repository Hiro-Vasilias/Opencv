//James Rogers Jan 2021 (c) Plymouth University
#include<iostream>
#include <fstream>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{

    VideoCapture InputStream("C:/AINT308Lib/Data/Task2 Video.mp4"); //Load in the video as an input stream
    const Point Pivot(592,52);                                      //Pivot position in the video

    //Open output file for angle data
    ofstream DataFile;
    DataFile.open ("C:/Users/zhuss/Desktop/AINT308Repository/Task2/Data.csv"); //comment this if you're testing this code
    //DataFile.open ("C:/AINT308Repository/Task2/Data.csv"); // Uncomment this if you have it saved at this location


    //loop through video frames
    while(true){

        //load next frame
        Mat Frame;
        InputStream.read(Frame);

        //if frame is empty then the video has ended, so break the loop
        if(Frame.empty()){
            break;
        }

        //video is very high resolution, reduce it to 720p to run faster
        resize(Frame,Frame,Size(1280,720));

        //======================================================Your code goes here====================================================
        //this code will run for each frame of the video. your task is to find the location of the swinging green target, and to find
        //its angle to the pivot. These angles will be saved to a .csv file where they can be plotted in Excel.


        //Convert RGB colour space to HSV
        Mat FrameHSV;
        cvtColor(Frame, FrameHSV, COLOR_BGR2HSV);

        //Filter required framed to a sub colour space (greyscale)
        Mat FrameFiltered;

        Vec3b LowerBound(45, 56, 60); // near black
        Vec3b UpperBound(96, 232, 210); // near green but not actual green.

        inRange(FrameHSV, LowerBound, UpperBound, FrameFiltered); // Frame filtered to greyscale using lower and upper bound limits

        //using Moments to calculate the angle of the pendulum as it swings.
        Moments m = moments(FrameFiltered, true);
        Point p(m.m10/m.m00, m.m01/m.m00); // Center point of the green rectangle on the pendulum.

        // finding pendulum center Marker Line endpoints
                Point center_vertical_Start(p.x,p.y-20);
                Point center_vertical_end(p.x,p.y+20);
                Point center_horizontal_Start(p.x-20,p.y);
                Point center_horizontal_end(p.x+20,p.y);

        // draw pivot  circles
                circle( Frame, Pivot,5, Scalar( 0, 0, 255 ), FILLED,LINE_8);
        // draw pendulum center
                circle( Frame,p,15, Scalar( 0, 255, 0 ), 2,LINE_8);
        // draw center cross
        line(Frame, center_vertical_Start,center_vertical_end,Scalar(0,255,0),2);
        line(Frame, center_horizontal_Start,center_horizontal_end,Scalar(0,255,0),2);

        // draw connecting line
        line(Frame, Pivot,p,Scalar(0,0,255),4);

        //Trignomentry to find angle
        float pi = 3.14159265358979323846; // Value of pi
        double Height = p.y - Pivot.y; // length of pendulum rod (red long line)
        double Distance = p.x - Pivot.x; // horizontal arc distance (not displacement)
        double Angle = atan2(Distance, Height) *(180 / pi); // use atan2 with arc & height to get angles in rads-1 and then convert to deg.


        cout<<"angle is "<< (int) Angle <<endl; // output Angle values to terminal
        DataFile << Angle << endl; // write to DataFile.csv

        //==============================================================================================================================

        //display the frame
        imshow("Video", Frame);
        imshow("Frame", FrameHSV);
        imshow("Filtered", FrameFiltered);
        waitKey(10);

            }

    DataFile.close(); //close output file
}
