//James Rogers Nov 2020 (c) Plymouth University
#include "main.h"
#include "faceclassifier.h"

/*
        ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
        ██░▄▄▄░█▄░▄█░██░█░▄▀█░▄▄█░▄▄▀█▄░▄███████▀░██░▄▄░█░▄▄█▀▄▄▀█▀▄▄▀█▀░██░▄▄░█▀░█
        ██▄▄▄▀▀██░██░██░█░█░█░▄▄█░██░██░█████████░██░▀▄░█▄▄▀█▀▄▄▀█▀▄▄▀██░██░▀▄░██░█
        ██░▀▀▀░██▄███▄▄▄█▄▄██▄▄▄█▄██▄██▄████████▀░▀█░▀▀░█▀▀▄█▄▀▀▄█▄▀▀▄█▀░▀█░▀▀░█▀░▀
        ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
*/

int main()
{

    //===========================Load Face Recognition Networks===========================
    String ClassNetPath = "C:/AINT308Lib/Data/Classifier Models/dlib_face_recognition_resnet_model_v1.dat";
    String GenderNetPath = "C:/AINT308Lib/Data/Classifier Models/dnn_gender_classifier_v1.dat";
    String AgeNetPath = "C:/AINT308Lib/Data/Classifier Models/dnn_age_predictor_v1.dat";

    FaceClassifier Classifier(ClassNetPath,GenderNetPath,AgeNetPath);

    //==============================Load Face Detector Model==============================
    String FaceDetectorPath = "C:/AINT308Lib/Data/Classifier Models/haarcascade_frontalface_default.xml";
    CascadeClassifier face_cascade;

    if(!face_cascade.load(FaceDetectorPath)){
            cout << "Error loading face cascade\n";
            return -1;
    }

    //===============================Find Profile Embeddings==============================
    Mat Profile = imread("C:/AINT308Lib/Data/Task5 Images/Profile.png"); //Load profile image

    //create a grey version of the profile picture
    Mat ProfileGray;
    cvtColor(Profile, ProfileGray, COLOR_BGR2GRAY);

    //find faces in the profile picture
    vector<Rect> ProfileFaces;
    face_cascade.detectMultiScale(ProfileGray, ProfileFaces);

    //if no faces are found, there has been an error in the facial detection
    if(ProfileFaces.size()==0){
        cout<<"No faces found in profile"<<endl;
        return -1;
    }

    //if a face/faces have been found, assume face[0] is the target and create a cropped image of it
    Mat ProfileFaceImg;
    Profile(ProfileFaces[0]).copyTo(ProfileFaceImg);

    //use the classifier class to extract facial embeddings
    dlib::matrix<float,0,1> ProfileEmbedding = Classifier.FaceEmbeddings(ProfileFaceImg);

    //draw a box around the target face
    rectangle(Profile, ProfileFaces[0], Scalar(0,255,0),2);

    //display the profile picture with the detected face untill x is pressed.
    while(waitKey(10)!='x'){
        imshow("Profile", Profile);
    }

    //================================Your code goes here=============================

    /*
            ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
            ██░▄▄▄░█▄░▄█░██░█░▄▀█░▄▄█░▄▄▀█▄░▄███████▀░██░▄▄░█░▄▄█▀▄▄▀█▀▄▄▀█▀░██░▄▄░█▀░█
            ██▄▄▄▀▀██░██░██░█░█░█░▄▄█░██░██░█████████░██░▀▄░█▄▄▀█▀▄▄▀█▀▄▄▀██░██░▀▄░██░█
            ██░▀▀▀░██▄███▄▄▄█▄▄██▄▄▄█▄██▄██▄████████▀░▀█░▀▀░█▀▀▄█▄▀▀▄█▄▀▀▄█▀░▀█░▀▀░█▀░▀
            ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
    */

    int count = 0;
    for (count =0; count<=4; count++){

        string path = "C:/AINT308Lib/Data/Task5 Images/"+to_string(count)+".png";
        Mat img = imread(path);

        // convert image to gray
        Mat Gray_Image;
        cvtColor(img, Gray_Image, COLOR_BGR2GRAY);

        // find faces in the image.
        vector<Rect> ImageFaces;
        face_cascade.detectMultiScale(Gray_Image, ImageFaces);

        // draw rectangles around faces

        if (ImageFaces.size()==0){
            cout<<"No Face found in image...."<<endl;
            return -1;
        }

        for(int i = 0; i<ImageFaces.size(); i++ ){
            rectangle(img, ImageFaces[i], Scalar(255,0,0),2);

        }
        vector<int> indices;
        vector<float> distances;

        float Threshold = 0.55;  // The threshold below which if the distance is, will be considered as Match.

        int MatchedFaceIndex = -1;
        // find the face embeddings of each face inside the image
        for(int j=0; j<ImageFaces.size(); j++){
               Mat faceImg;
               img(ImageFaces[j]).copyTo(faceImg);

               // get each Face Embedding
                dlib::matrix<float,0,1> EachFaceEmbedding = Classifier.FaceEmbeddings(faceImg);

                float distance;
                distance = dlib::length(ProfileEmbedding-EachFaceEmbedding);

                cout<<"DISTANCE/DIFFERENCE FROM PROFILE FACE OF FACE "<<j<<" is : "<<distance<<endl;
                if (distance<Threshold){
                    cout<<"FOUND MATCHED FACE : "<<j<<endl;
                    MatchedFaceIndex = j;
                    indices.push_back(j);
                    distances.push_back(distance);
                }

        }

        /*
                ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
                ██░▄▄▄░█▄░▄█░██░█░▄▀█░▄▄█░▄▄▀█▄░▄███████▀░██░▄▄░█░▄▄█▀▄▄▀█▀▄▄▀█▀░██░▄▄░█▀░█
                ██▄▄▄▀▀██░██░██░█░█░█░▄▄█░██░██░█████████░██░▀▄░█▄▄▀█▀▄▄▀█▀▄▄▀██░██░▀▄░██░█
                ██░▀▀▀░██▄███▄▄▄█▄▄██▄▄▄█▄██▄██▄████████▀░▀█░▀▀░█▀▀▄█▄▀▀▄█▄▀▀▄█▀░▀█░▀▀░█▀░▀
                ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
        */


        cout<<"Matched face Index is "<<MatchedFaceIndex<<"for an Image countber"<<count<<endl;

        if (MatchedFaceIndex!=-1){
            rectangle(img, ImageFaces[MatchedFaceIndex], Scalar(0,255,0),4);

        }

        printf("press x key to go next image");

        imshow("Image",img);

        waitKey(-1);


    }

    /*
            ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
            ██░▄▄▄░█▄░▄█░██░█░▄▀█░▄▄█░▄▄▀█▄░▄███████▀░██░▄▄░█░▄▄█▀▄▄▀█▀▄▄▀█▀░██░▄▄░█▀░█
            ██▄▄▄▀▀██░██░██░█░█░█░▄▄█░██░██░█████████░██░▀▄░█▄▄▀█▀▄▄▀█▀▄▄▀██░██░▀▄░██░█
            ██░▀▀▀░██▄███▄▄▄█▄▄██▄▄▄█▄██▄██▄████████▀░▀█░▀▀░█▀▀▄█▄▀▀▄█▄▀▀▄█▀░▀█░▀▀░█▀░▀
            ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
    */
















}
















