//10588101 feb 2021 (c) Plymouth University
#include <iostream>

#include<opencv2/opencv.hpp>
#include<opencv2/opencv_modules.hpp>

using namespace std;
using namespace cv;

int main(){

    //Path of image folder
    string PathToFolder = "C:/AINT308Lib/Data/Task1 Images/";

    //Loop through the 30 car images
    for(int n=0; n<30; ++n){

        //Each image is named 0.png, 1.png, 2.png, etc. So generate the image file path based on n and the folder path
        string PathToImage = PathToFolder+to_string(n)+".png";

        //Load car image at the file paths location
        Mat Car=imread(PathToImage);

        //Your code goes here. The example code below shows you how to read the red, green, and blue colour values of the
        //pixel at position (0,0). Modify this section to check not just one pixel, but all of them in the 640x480 image
        //(using for-loops), and using the RGB values classifiy if a given pixel looks red, green, blue, or other. For
        //example, if the blue value is 30, green is 10, and red is 255, there is a lot more red in the pixel than the
        //other two colours so it will look red. Now count all the red, green, and blue pixels in your image, and classify
        //what colour the car is. for example if you count 30000 blue pixels, 100 green pixels, and 2000 red pixels, the
        //car is probably blue as the image contains a lot more blue than the other colours.

        //==============example code, feel free to delete=============
        int R_Count =0, G_Count =0, B_Count =0;                //Initial dynamic storage values
        int OLD_R_Count =0, OLD_G_Count=0, OLD_B_Count=0;      //accumulating counter storage values


        for(int x=0; x<640; ++x){                              // loop for the X-axis
            for (int y=0; y<480; ++y){                         // loop for the Y-axis

                Vec3b PixelValue = Car.at<Vec3b>(y,x);         // Load pixel values
                int blue=PixelValue[0], green=PixelValue[1], red=PixelValue[2]; //Fully optional, it's an extra line of code to make it easier

                R_Count = red + OLD_R_Count;                   // read pixel value and store
                G_Count = green + OLD_G_Count;                 // add accumulating pixel value and store
                B_Count = blue + OLD_B_Count;

                OLD_R_Count = R_Count;                         // load initial dynamic counter values into accumulating counter storage.
                OLD_G_Count = G_Count;
                OLD_B_Count = B_Count;
            }
        }

//        cout<<"The blue value is " <<(int)OLD_B_Count<<endl;     // Uncomment for Diagnostic
//        cout<<"The green value is "<<(int)OLD_G_Count<<endl;     // Will output each RGB Values
//        cout<<"The red value is "  <<(int)OLD_R_Count<<endl;
//        cout<<"                 "  <<endl;

        if (OLD_R_Count > OLD_G_Count && OLD_R_Count > OLD_B_Count){cout<<"The Car is Red"<<endl;}   //Represent most pixel value as colour of the image
        if (OLD_B_Count > OLD_R_Count && OLD_B_Count > OLD_G_Count){cout<<"The Car is Blue"<<endl;}  //Most pixel value is output as colour name
        if (OLD_G_Count > OLD_R_Count && OLD_G_Count > OLD_B_Count){cout<<"The Car is Green"<<endl;}



        //============================================================

        //display the car image untill x is pressed
        while(waitKey(10)!='x'){
            imshow("Car", Car);
        }

    }

}




















