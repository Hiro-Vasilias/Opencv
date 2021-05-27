//Hasnain Shah May 2021 (c) Plymouth University
#include<iostream>
#include <fstream>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int solution_number=0;

/*
        ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
        ██░▄▄▄░█▄░▄█░██░█░▄▀█░▄▄█░▄▄▀█▄░▄███████▀░██░▄▄░█░▄▄█▀▄▄▀█▀▄▄▀█▀░██░▄▄░█▀░█
        ██▄▄▄▀▀██░██░██░█░█░█░▄▄█░██░██░█████████░██░▀▄░█▄▄▀█▀▄▄▀█▀▄▄▀██░██░▀▄░██░█
        ██░▀▀▀░██▄███▄▄▄█▄▄██▄▄▄█▄██▄██▄████████▀░▀█░▀▀░█▀▀▄█▄▀▀▄█▄▀▀▄█▀░▀█░▀▀░█▀░▀
        ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
*/

/*Function for the Canny edge detector */
Mat canny_edge_detector(Mat Image)
{

    // Convert the Frame color to grayscale
    Mat gray_Frame;
    cvtColor(Image, gray_Frame, COLOR_RGB2GRAY);

    //Extract yellow and white info
    Mat maskYellow, maskWhite;

    inRange(gray_Frame, Scalar(20, 100, 100), Scalar(30, 255, 255), maskYellow);
    inRange(gray_Frame, Scalar(150, 150, 150), Scalar(255, 255, 255), maskWhite);

    Mat mask, processed;
    bitwise_or(maskYellow, maskWhite, mask); //Combine the two masks
    bitwise_and(gray_Frame, mask, processed); //Extract what matches

    // Reduce noise from the Frame
    Mat Blur;
    GaussianBlur(processed, Blur, Size(5, 5), 0);

    //fill the gaps
    Mat kernel = Mat::ones(15, 15, CV_8U);
    dilate(Blur, Blur, kernel);
    erode(Blur, Blur, kernel);
    morphologyEx(Blur, Blur, MORPH_CLOSE, kernel);

    Mat canny;
    Canny(Blur, canny, 50, 150, 3);
    return canny;
}
/*Function for the Hough Lines */
Mat HoughLP(Mat Og_Frame, Mat canny_whiteline, int x)
{

    Mat With_HoughLines = Og_Frame;
    Mat Probab_HoughLines = canny_whiteline;
        // Standard Hough Line Transform
        vector<Vec2f> lines; // will hold the results of the detection
    HoughLines(canny_whiteline, lines, 1, CV_PI/180,x, 0, 0); // runs the actual detection
    //draw lines
    for( size_t i = 0; i < lines.size(); i++ )
        {
            float rho = lines[i][0], theta = lines[i][1];
            Point pt1, pt2;
            double a = cos(theta), b = sin(theta);
            double x0 = a*rho, y0 = b*rho;
            pt1.x = cvRound(x0 + 1000*(-b));
            pt1.y = cvRound(y0 + 1000*(a));
            pt2.x = cvRound(x0 - 1000*(-b));
            pt2.y = cvRound(y0 - 1000*(a));
            line( With_HoughLines, pt1, pt2, Scalar(255,0,0), 6, LINE_AA);
        }

    // Probabilistic Line Transform
        vector<Vec4i> linesP; // will hold the results of the detection
        HoughLinesP(canny_whiteline, linesP, 1, CV_PI/180, 190, 100, 100 ); // runs the actual detection
        // Draw the lines
        for( size_t i = 0; i < linesP.size(); i++ )
        {
            Vec4i l = linesP[i];
            line( Probab_HoughLines, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, LINE_AA);
        }

    //return Probab_HoughLines;
        return With_HoughLines;
}
//unused Function for Region of interest, experimental code
Mat Roi2(Mat input_image)
{
    Mat Image = input_image;
    Mat mask = Mat::zeros(Image.size(),CV_8UC3);
    Mat Final = Mat::zeros(Image.size(),CV_8UC1);
    Scalar white = Scalar(255,255,255);

    vector<Point> pts;
    Point pt0 = Point(0,500);                             //Top Left corner
    Point pt1 = Point(int(Image.size().width)/2280);      //Bottom Left Corner
    Point pt2 = Point(1140,(int(Image.size().height)));   //Bottom Right corner
    Point pt3 = Point(0,(int(Image.size().height)));      //Top Right Corner
    pts.push_back(pt0);
    pts.push_back(pt1);
    pts.push_back(pt2);
    pts.push_back(pt3);

    fillConvexPoly(mask,pts,white,8,0);
    bitwise_and(Image,Image,Final,mask);










    return(Final);

}
//Sliding window function for the movement of the lane window using inverse Perspective to draw shapes
vector<Point2f> slidingWindow(Mat image, Rect window)
{
    vector<Point2f> points;
    const Size imgSize = image.size();
    bool shouldBreak = false;

    while (true)
    {
        float currentX = window.x + window.width * 0.5f;

        Mat roi = image(window); //Extract region of interest
        vector<Point2f> locations;

        findNonZero(roi, locations); //Get all non-black pixels. All are white in our case
        float avgX = 0.0f;

        for (int i = 0; i < locations.size(); ++i) //Calculate average X position
        {
            float x = locations[i].x;
            avgX += window.x + x;
        }

        avgX = locations.empty() ? currentX : avgX / locations.size();

        Point point(avgX, window.y + window.height * 0.5f);
        points.push_back(point);

        //Move the window up
        window.y -= window.height;

        //For the uppermost position
        if (window.y < 0)
        {
            window.y = 0;
            shouldBreak = true;
        }

        //Move x position
        window.x += (point.x - currentX);

        //Make sure the window doesn't overflow, we get an error if we try to get data outside the matrix
        if (window.x < 0)
            window.x = 0;
        if (window.x + window.width >= imgSize.width)
            window.x = imgSize.width - window.width - 1;

        if (shouldBreak)
            break;
    }

    return points;
}
//Function that defines the ROI and the bird view for Sliding window function, Also draws the road overlay
Mat region_of_interest(Mat image)
{

    //Define points that are used for generating bird's eye view. This was done by trial and error. Best to prepare sliders and configure for each use case.
    Point2f srcVertices[4];              //#1         //#2          //#3
    srcVertices[0] = Point(500, 425);    //560,425    //500,425     //000,720
    srcVertices[1] = Point(830, 425);    //720,425    //830,425     //720,720
    srcVertices[2] = Point(1400, 1030);  //1400,1030  //1400,1030   //000,1280
    srcVertices[3] = Point(010, 1030);   //10, 1030   //10,1030     //2200,1280

    // original resolution 1280 x 720p
    //Destination vertices. Output is 640 by 480px
    Point2f dstVertices[4];
    dstVertices[0] = Point(0, 0);
    dstVertices[1] = Point(640, 0);    //#3 720,0
    dstVertices[2] = Point(640, 480); //#3 720,1280
    dstVertices[3] = Point(0, 480);   //#3 0,1280

    //Prepare matrix for transform and get the warped image
    Mat perspectiveMatrix = getPerspectiveTransform(srcVertices, dstVertices);
    Mat dst(480, 640, CV_8UC3); //Destination for warped image  //#3 720,1280
    //For transforming back into original image space
    Mat invertedPerspectiveMatrix;
    invert(perspectiveMatrix, invertedPerspectiveMatrix);

    warpPerspective(image, dst, perspectiveMatrix, dst.size(), INTER_LINEAR, BORDER_CONSTANT);

    Mat Canny_Frame = canny_edge_detector(dst);

    threshold(Canny_Frame, Canny_Frame, 160, 255, THRESH_BINARY);

    vector<Point2f> pts = slidingWindow(Canny_Frame, Rect(0, 420, 120, 60));


    vector<Point> allPts; //Used for the end polygon at the end.
    vector<Point2f> outPts;
    perspectiveTransform(pts, outPts, invertedPerspectiveMatrix); //Transform points back into original image space
    //Draw the points onto the out image
    for (int i = 0; i < outPts.size() - 1; ++i)
    {
        line(image, outPts[i], outPts[i + 1], Scalar(255, 0, 0), 3);
        allPts.push_back(Point(outPts[i].x, outPts[i].y));
    }

    allPts.push_back(Point(outPts[outPts.size() - 1].x, outPts[outPts.size() - 1].y));

    Mat out;

    cvtColor(Canny_Frame, out, COLOR_GRAY2BGR); //Conver the processing image to color so that we can visualise the lines
    for (int i = 0; i < pts.size() - 1; ++i) //Draw a line on the processed image
        line(out, pts[i], pts[i + 1], Scalar(255, 0, 0));

    //Sliding window for the right side
    pts = slidingWindow(Canny_Frame, Rect(520, 420, 120, 60));
    perspectiveTransform(pts, outPts, invertedPerspectiveMatrix);

    //Draw the other lane and append points
    for (int i = 0; i < outPts.size() - 1; ++i)
    {
        line(image, outPts[i], outPts[i + 1], Scalar(0, 0, 255), 3);
        allPts.push_back(Point(outPts[outPts.size() - i - 1].x, outPts[outPts.size() - i - 1].y));
    }

    allPts.push_back(Point(outPts[0].x - (outPts.size() - 1) , outPts[0].y));

    for (int i = 0; i < pts.size() - 1; ++i)
        line(out, pts[i], pts[i + 1], Scalar(0, 0, 255));

    //Create a purple-ish overlay
    vector<vector<Point>> arr;
    arr.push_back(allPts);
    Mat overlay = Mat::zeros(image.size(), image.type());
    fillPoly(overlay, arr, Scalar(190, 63, 102));
    addWeighted(image, 1, overlay, 0.5, 0, image); //Overlay it
    imshow("Output", image);


return image;
}

/*
        ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
        ██░▄▄▄░█▄░▄█░██░█░▄▀█░▄▄█░▄▄▀█▄░▄███████▀░██░▄▄░█░▄▄█▀▄▄▀█▀▄▄▀█▀░██░▄▄░█▀░█
        ██▄▄▄▀▀██░██░██░█░█░█░▄▄█░██░██░█████████░██░▀▄░█▄▄▀█▀▄▄▀█▀▄▄▀██░██░▀▄░██░█
        ██░▀▀▀░██▄███▄▄▄█▄▄██▄▄▄█▄██▄██▄████████▀░▀█░▀▀░█▀▀▄█▄▀▀▄█▄▀▀▄█▀░▀█░▀▀░█▀░▀
        ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
*/

// Experimental Function to calculate ROI with Hough Lines
Mat ROI_Hough(Mat Image)
{
    Mat mask = Mat::zeros(Image.size(),CV_8UC3);

    int height = Image.size().height;
    int width = Image.size().width;
/*
 * (0,0)---------------------------->
 * |
 * |       Top_Left ------- Top_Right
 * |       /                       \
 * V      /                         \
 *       /                           \
 *   Bottom_Left --------------Bottom_Right
 */

    //Define points that are used for generating bird's eye view. This was done by trial and error. Best to prepare sliders and configure for each use case.
    Point2f srcVertices[4];              //#1         //#2          //#3
    srcVertices[0] = Point(500, height/2.1);    //560,425    //500,425     //000,720
    srcVertices[1] = Point(720, height/2.1);    //Top Right (X,Y)
    srcVertices[2] = Point(1200, width/2);      //Bottom Right
    srcVertices[3] = Point(700, width/2);   //10, 1030   //10,1030     //2200,1280

    // original resolution 1280 x 720p
    //Destination vertices. Output is 640 by 480px
    Point2f dstVertices[4];
    dstVertices[0] = Point(0, 0);
    dstVertices[1] = Point(width, 0);    //#3 720,0
    dstVertices[2] = Point(width, height); //#3 720,1280
    dstVertices[3] = Point(0, width);   //#3 0,1280

    //Prepare matrix for transform and get the warped image
    Mat perspectiveMatrix = getPerspectiveTransform(srcVertices, dstVertices);
    //Mat dst(480, 640, CV_8UC3); //Destination for warped image  //#3 720,1280
    //For transforming back into original image space
    Mat invertedPerspectiveMatrix;
    invert(perspectiveMatrix, invertedPerspectiveMatrix);

    warpPerspective(Image, mask, perspectiveMatrix, mask.size(), INTER_LINEAR, BORDER_CONSTANT);

    Mat Canny_Frame = canny_edge_detector(mask);

    threshold(Canny_Frame, Canny_Frame, 160, 255, THRESH_BINARY);
    imshow("canny",mask);

//    Mat HP = HoughLP(Image,Canny_Frame);
//    imshow("hp",HP);


    vector<Point2f> pts = slidingWindow(Canny_Frame, Rect(0, 420, 120, 60));

    vector<Point> allPts; //Used for the end polygon at the end.
    vector<Point2f> outPts;
    perspectiveTransform(pts, outPts, invertedPerspectiveMatrix); //Transform points back into original image space

    return (Image);

}
// Function for Image filteration to get HSV to Hough Lines
Mat Filtered_Hough(Mat image)
{

    Mat g1,g2;
    GaussianBlur(image,g1,Size(9,9),0);
    GaussianBlur(image,g2,Size(9*5,9*5),0);
    Mat result = (g1-g2)*2;
    imshow("dog",result);
    Mat HSV, green, FrameFiltered;
    cvtColor(result,HSV, COLOR_BGR2HSV);

    Vec3b LB(50,2,0);
    Vec3b UH(132,16,105);
    inRange(HSV,LB , UH, FrameFiltered);
    Canny(FrameFiltered,green,50,150,3);
    //cvtColor(FrameFiltered,green,COLOR_HSV2RGB);
    Mat With_HoughLines = HoughLP(image, green, 110);

    //imshow("withHoughlines",With_HoughLines);
    //imshow("1",HSV);
    //imshow("2",green);
    //imshow("3",FrameFiltered);

    return(With_HoughLines);
}
//sequence threads with explaination within
void sequence0(Mat Frame)
{
    //Image is send to Filtered_Hough to get a HSV > hough lines
    //this is then perspective warped to get bird view and then inversed to the original image
    //to allow road drawings
Mat Unorthodox_Hough = Filtered_Hough(Frame);
Mat warped_roi = region_of_interest(Frame);


}
//sequence threads with explaination within
void sequence1(Mat Frame)
{
    //This function calls in the canny_edge_detector and then uses Houghlines to calculate the amount of straight lines
    //meeting the same intersection line point.

    //Run canny_edge_detector function
    Mat Canny_Frame = canny_edge_detector(Frame);
    imshow("edgy", Canny_Frame);

    Mat With_HoughLines = HoughLP(Frame, Canny_Frame, 230);
    imshow("withHoughlines",With_HoughLines);

}
//sequence threads with explaination within
void sequence2(Mat Frame)
{
      //This Line of code calls into the region_of_interest function
    Mat warped_roi = region_of_interest(Frame);
      //This then calls in canny_edge_detector after allocating ROI points.
      //after processing canny it calls in slidingWindow function multiple times
      //inverses the processed ROI into original image and applies drawings.

      //Result should be a green overlay on the road showing the path the vehicle is going on
      //This was done via a top down bird view perspective.


}


/*
        ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
        ██░▄▄▄░█▄░▄█░██░█░▄▀█░▄▄█░▄▄▀█▄░▄███████▀░██░▄▄░█░▄▄█▀▄▄▀█▀▄▄▀█▀░██░▄▄░█▀░█
        ██▄▄▄▀▀██░██░██░█░█░█░▄▄█░██░██░█████████░██░▀▄░█▄▄▀█▀▄▄▀█▀▄▄▀██░██░▀▄░██░█
        ██░▀▀▀░██▄███▄▄▄█▄▄██▄▄▄█▄██▄██▄████████▀░▀█░▀▀░█▀▀▄█▄▀▀▄█▄▀▀▄█▀░▀█░▀▀░█▀░▀
        ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
*/

int main()
{
    //Change the number for the solution to play
    solution_number = 3;

    VideoCapture InputStream("C:/AINT308Lib/Data/Proposal5 Video.mp4"); //Load in the video as an input stream

    cout<<"Press any key to exit";

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

        imshow("Original Frames", Frame);
        if (waitKey(50) > 0)
           break;
        //Frame.release();

        //------------------------------------------------------

        switch ((int)solution_number)
        {
        case 0 :
            sequence0(Frame); //Custom filtered Gaussian blur > HSV > Canny > HoughLine
            sequence2(Frame); //Rendering using ROI and adding Lane overlay
        case 1 :
            sequence1(Frame); //canny to HoughLine
            sequence2(Frame); //Rendering using ROI and adding Lane overlay
        case 2 :
            sequence1(Frame); //Canny to HoughLine output //output is only straight lines
        case 3 :
            sequence2(Frame); //Rendering using ROI and adding Lane overlay
        }
    }

}
/*
        ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
        ██░▄▄▄░█▄░▄█░██░█░▄▀█░▄▄█░▄▄▀█▄░▄███████▀░██░▄▄░█░▄▄█▀▄▄▀█▀▄▄▀█▀░██░▄▄░█▀░█
        ██▄▄▄▀▀██░██░██░█░█░█░▄▄█░██░██░█████████░██░▀▄░█▄▄▀█▀▄▄▀█▀▄▄▀██░██░▀▄░██░█
        ██░▀▀▀░██▄███▄▄▄█▄▄██▄▄▄█▄██▄██▄████████▀░▀█░▀▀░█▀▀▄█▄▀▀▄█▄▀▀▄█▀░▀█░▀▀░█▀░▀
        ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
*/
