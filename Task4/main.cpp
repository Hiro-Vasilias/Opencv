//James Rogers Nov 2020 (c) Plymouth University
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <stdio.h>

using namespace cv;
using namespace std;

Mat getDistanceMap(Mat disp_Map){
    Mat dist1Map;
    // Formula to convert disparity into Distance
    //dist = (int)3664 / pixel;
    dist1Map = disp_Map;
    for(int i=0; i<disp_Map.rows; i++){
        for(int j=0; j<disp_Map.cols; j++){
            //distMap.at<cv::Vec3b>(i,j) = (int)3664 / disp_Map.at<cv::Vec3b>(i,j);
            int intensity = disp_Map.at<uchar>(i, j);

            if (intensity!=0){
                    //cout<<"intensity"<<intensity<<endl;
                    float dist_value = 3664 / intensity;
                    int dist_val_int = (int)dist_value;

                    dist1Map.at<uchar>(i, j) = dist_val_int;
                    //cout<<"distMap.at<uchar>(i, j)"<<distMap.at<uchar>(j, i)<<endl;
            }

        }

    }

    return dist1Map;
}

/*/////////////////////////////////////////////////////////////////////////////////////
▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
██░▄▄▄░█▄░▄█░██░█░▄▀█░▄▄█░▄▄▀█▄░▄███████▀░██░▄▄░█░▄▄█▀▄▄▀█▀▄▄▀█▀░██░▄▄░█▀░█
██▄▄▄▀▀██░██░██░█░█░█░▄▄█░██░██░█████████░██░▀▄░█▄▄▀█▀▄▄▀█▀▄▄▀██░██░▀▄░██░█
██░▀▀▀░██▄███▄▄▄█▄▄██▄▄▄█▄██▄██▄████████▀░▀█░▀▀░█▀▀▄█▄▀▀▄█▄▀▀▄█▀░▀█░▀▀░█▀░▀
▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
/////////////////////////////////////////////////////////////////////////////////////*/

int main(int argc, char** argv)
{
    //Calibration file paths (you need to make these)
    string intrinsic_filename = "C:/AINT308Lib/Data/intrinsics.xml";
    string extrinsic_filename = "C:/AINT308Lib/Data/extrinsics.xml";

    //================================================Load Calibration Files===============================================
    //This code loads in the intrinsics.xml and extrinsics.xml calibration files, and creates: map11, map12, map21, map22.
    //These four matrices are used to distort the camera images to apply the lense correction.
    Rect roi1, roi2;
    Mat Q;
    Size img_size = {640,480};

    FileStorage fs(intrinsic_filename, FileStorage::READ);
    if(!fs.isOpened()){
        printf("Failed to open file %s\n", intrinsic_filename.c_str());
        return -1;
    }

    Mat M1, D1, M2, D2;
    fs["M1"] >> M1;
    fs["D1"] >> D1;
    fs["M2"] >> M2;
    fs["D2"] >> D2;

    fs.open(extrinsic_filename, FileStorage::READ);
    if(!fs.isOpened())
    {
        printf("Failed to open file %s\n", extrinsic_filename.c_str());
        return -1;
    }
    Mat R, T, R1, P1, R2, P2;
    fs["R"] >> R;
    fs["T"] >> T;

    stereoRectify( M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2 );

    Mat map11, map12, map21, map22;
    initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
    initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);

    //===============================================Stereo SGBM Settings==================================================
    //This sets up the block matcher, which is used to create the disparity map. The various settings can be changed to
    //obtain different results. Note that some settings will crash the program.

    int SADWindowSize=3;            //must be an odd number >=3
    int numberOfDisparities=272;    //must be divisable by 16

    Ptr<StereoSGBM> sgbm = StereoSGBM::create(0,16,3);
    sgbm->setBlockSize(SADWindowSize);
    sgbm->setPreFilterCap(55);
    sgbm->setP1(8*3*SADWindowSize*SADWindowSize);
    sgbm->setP2(32*3*SADWindowSize*SADWindowSize);
    sgbm->setMinDisparity(0);
    sgbm->setNumDisparities(numberOfDisparities);
    sgbm->setUniquenessRatio(10);
    sgbm->setSpeckleWindowSize(150);
    sgbm->setSpeckleRange(32);
    sgbm->setDisp12MaxDiff(1);
    sgbm->setMode(StereoSGBM::MODE_SGBM);

    //==================================================Main Program Loop================================================
    int ImageNum=0; //current image index
    int Imagedist=30; //current image distance index
    while (1){

        //==================================Your code goes here===============================

        // Set Disparity Image for Known distance targets
        //Load images from file (needs changing for known distance targets)
        Mat Left =imread("C:/AINT308Lib/Data/Task4 Distance Targets/left" +to_string(Imagedist)+"cm.jpg");
        Mat Right=imread("C:/AINT308Lib/Data/Task4 Distance Targets/right"+to_string(Imagedist)+"cm.jpg");
        cout<<"Loaded image: "<<Imagedist<<endl;
        //Distort image to correct for lens/positional distortion
        remap(Left, Left, map11, map12, INTER_LINEAR);
        remap(Right, Right, map21, map22, INTER_LINEAR);

        //Match left and right images to create disparity image
        Mat disp16bit, disp8bit;
        sgbm->compute(Left, Right, disp16bit);                               //compute 16-bit greyscalse image with the stereo block matcher
        disp16bit.convertTo(disp8bit, CV_8U, 255/(numberOfDisparities*16.)); //Convert disparity map to an 8-bit greyscale image so it can be displayed (imshow only works with 8bit images)


        // Setting distance calibration from unknown distance targets
        // A. Set disparity image of unknown distance targets.

        //Load images from file (needs changing for known distance targets)
        Mat Left_Target =imread("C:/AINT308Lib/Data/Task4 Unknown Targets/left" +to_string(ImageNum)+".jpg");
        Mat Right_Target=imread("C:/AINT308Lib/Data/Task4 Unknown Targets/right"+to_string(ImageNum)+".jpg");
        cout<<"Loaded image: "<<ImageNum<<endl;

        //Distort image to correct for lens/positional distortion
        remap(Left_Target, Left_Target, map11, map12, INTER_LINEAR);
        remap(Right_Target, Right_Target, map21, map22, INTER_LINEAR);

        //Match left and right images to create disparity image
        Mat disp16bit_Target, disp8bit_Target;
        sgbm->compute(Left_Target, Right_Target, disp16bit_Target);                               //compute 16-bit greyscalse image with the stereo block matcher
        disp16bit_Target.convertTo(disp8bit_Target, CV_8U, 255/(numberOfDisparities*16.)); //Convert disparity map to an 8-bit greyscale image so it can be displayed (imshow only works with 8bit images)

        //Data written to excel manually by manually comparing known distance to target distance


        // B. create a disparity distance map
        // using formula distance = Focal x disparity / pixel
        // see excel sheet "Task 4 - Disparity Mapping Results.xlsx" for reference
        // Distance = 3840 / Pixel

        Mat distMap, disp8bitCopy;
        disp8bitCopy = disp8bit_Target.clone();
        distMap = getDistanceMap(disp8bitCopy);

        //

/*
        ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
        ██░▄▄▄░█▄░▄█░██░█░▄▀█░▄▄█░▄▄▀█▄░▄███████▀░██░▄▄░█░▄▄█▀▄▄▀█▀▄▄▀█▀░██░▄▄░█▀░█
        ██▄▄▄▀▀██░██░██░█░█░█░▄▄█░██░██░█████████░██░▀▄░█▄▄▀█▀▄▄▀█▀▄▄▀██░██░▀▄░██░█
        ██░▀▀▀░██▄███▄▄▄█▄▄██▄▄▄█▄██▄██▄████████▀░▀█░▀▀░█▀▀▄█▄▀▀▄█▄▀▀▄█▀░▀█░▀▀░█▀░▀
        ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
*/




        //display images untill x is pressed
        //   int key=0;
        while(waitKey(10)!='x')
        {
            imshow("left", Left);
            imshow("right", Right);
            imshow("disparity", disp8bit);

            imshow("left unknown distance", Left_Target);
            imshow("right unknown distance", Right_Target);
            imshow("disparity unknown", disp8bit_Target);

            imshow("distMap", distMap);

            //imshow("disparity-distance map", distance_map);
        };

        //move to next image
        ImageNum++;
        Imagedist=Imagedist+10;
        if (Imagedist>150)
        {
            Imagedist=30;
        }
        if(ImageNum>7)
        {
            ImageNum=0;
        }
    }

    return 0;
}



