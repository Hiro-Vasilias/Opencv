//James Rogers Jan 2021 (c) Plymouth University
#include<iostream>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{

    //Path to image file
    string Path = "C:/AINT308Lib/Data/Task3 Images/";

    //loop through component images
    for(int n=0; n<10; ++n){

        //read PCB and component images
        Mat PCB = imread(Path+"PCB.png");
        Mat Component = imread(Path+"Component"+to_string(n)+".png");

        //================Your code goes here=====================

        Mat Mask;
        matchTemplate( PCB, Component, Mask, TM_SQDIFF_NORMED);

        double Min_Val, Max_Val;
        Point Min_Loc, Max_Loc;

        minMaxLoc(Mask, &Min_Val, &Max_Val, &Min_Loc, &Max_Loc); // find the min location and max locational values
        Point Match_Loc = Min_Loc; // since TM_SQDIFF_NORMED's best matches are lowest values.

        int Bottom_right_x = Match_Loc.x + Component.cols; // Bottom right corner point x-axis
        int Bottom_right_y = Match_Loc.y + Component.rows; // bottom right corner point y-axis

        // draw a rectangle around the identified component
        rectangle(PCB, Match_Loc, Point(Bottom_right_x, Bottom_right_y), Scalar(0,0,255), 2,8,0);

        //--------------------------------------------------------
        //display the results untill x is pressed
        while(waitKey(10)!='x'){
            imshow("Target", Component);
            imshow("PCB", PCB);
            imshow("output", Mask);
        }

    }

}
