#ifndef FACECLASSIFIER_H
#define FACECLASSIFIER_H

/*
Code developed by James Rogers to support AINT308
University of Plymouth
james.rogers@plymouth.ac.uk
*/

#include "main.h"


//===============================================Classifier Template============================================================
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = dlib::add_prev1<block<N,BN,1,dlib::tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = dlib::add_prev2<dlib::avg_pool<2,2,2,2,dlib::skip1<dlib::tag2<block<N,BN,2,dlib::tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<dlib::con<N,3,3,1,1,dlib::relu<BN<dlib::con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = dlib::relu<residual<block,N,dlib::affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = dlib::relu<residual_down<block,N,dlib::affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = dlib::loss_metric<dlib::fc_no_bias<128,dlib::avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            dlib::max_pool<3,3,2,2,dlib::relu<dlib::affine<dlib::con<32,7,7,2,2,
                            dlib::input_rgb_image_sized<150>
                            >>>>>>>>>>>>;

//==================================================Gender Classifier Type==================================================

template <int N, template <typename> class BN, int stride, typename SUBNET>
using gblock = BN<dlib::con<N, 3, 3, stride, stride, dlib::relu<BN<dlib::con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET> using res_ = dlib::relu<gblock<N, dlib::bn_con, 1, SUBNET>>;
template <int N, typename SUBNET> using ares_ = dlib::relu<gblock<N, dlib::affine, 1, SUBNET>>;

template <typename SUBNET> using galevel1 = dlib::avg_pool<2, 2, 2, 2, ares_<64, SUBNET>>;
template <typename SUBNET> using galevel2 = dlib::avg_pool<2, 2, 2, 2, ares_<32, SUBNET>>;

using agender_type = dlib::loss_multiclass_log<dlib::fc<2, dlib::multiply<dlib::relu<dlib::fc<16, dlib::multiply<galevel1<galevel2< dlib::input_rgb_image_sized<32>>>>>>>>>;

//==================================================Age Classifier Type==================================================

// This block of statements defines a Resnet-10 architecture for the age predictor.
// We will use 81 classes (0-80 years old) to predict the age of a face.
const unsigned long number_of_age_classes = 81;

// The resnet basic block.
template<
    int num_filters,
    template<typename> class BN,  // some kind of batch normalization or affine layer
    int stride,
    typename SUBNET
>
using basicblock = BN<dlib::con<num_filters, 3, 3, 1, 1, dlib::relu<BN<dlib::con<num_filters, 3, 3, stride, stride, SUBNET>>>>>;

// A residual making use of the skip layer mechanism.
template<
    template<int, template<typename> class, int, typename> class BLOCK,  // a basic block defined before
    int num_filters,
    template<typename> class BN,  // some kind of batch normalization or affine layer
    typename SUBNET
> // adds the block to the result of tag1 (the subnet)
using residual = dlib::add_prev1<BLOCK<num_filters, BN, 1, dlib::tag1<SUBNET>>>;

// A residual that does subsampling (we need to subsample the output of the subnet, too).
template<
    template<int, template<typename> class, int, typename> class BLOCK,  // a basic block defined before
    int num_filters,
    template<typename> class BN,
    typename SUBNET
>
using residual_down = dlib::add_prev2<dlib::avg_pool<2, 2, 2, 2, dlib::skip1<dlib::tag2<BLOCK<num_filters, BN, 2, dlib::tag1<SUBNET>>>>>>;

// Residual block with optional downsampling and batch normalization.
template<
    template<template<int, template<typename> class, int, typename> class, int, template<typename>class, typename> class RESIDUAL,
    template<int, template<typename> class, int, typename> class BLOCK,
    int num_filters,
    template<typename> class BN,
    typename SUBNET
>
using residual_block = dlib::relu<RESIDUAL<BLOCK, num_filters, BN, SUBNET>>;

template<int num_filters, typename SUBNET>
using aresbasicblock_down = residual_block<residual_down, basicblock, num_filters, dlib::affine, SUBNET>;

// Some useful definitions to design the affine versions for inference.
template<typename SUBNET> using aresbasicblock256 = residual_block<residual, basicblock, 256, dlib::affine, SUBNET>;
template<typename SUBNET> using aresbasicblock128 = residual_block<residual, basicblock, 128, dlib::affine, SUBNET>;
template<typename SUBNET> using aresbasicblock64  = residual_block<residual, basicblock, 64, dlib::affine, SUBNET>;

// Common input for standard resnets.
template<typename INPUT>
using aresnet_input = dlib::max_pool<3, 3, 2, 2, dlib::relu<dlib::affine<dlib::con<64, 7, 7, 2, 2, INPUT>>>>;

// Resnet-10 architecture for estimating.
template<typename SUBNET>
using aresnet10_level1 = aresbasicblock256<aresbasicblock_down<256, SUBNET>>;
template<typename SUBNET>
using aresnet10_level2 = aresbasicblock128<aresbasicblock_down<128, SUBNET>>;
template<typename SUBNET>
using aresnet10_level3 = aresbasicblock64<SUBNET>;
// The resnet 10 backbone.
template<typename INPUT>
using aresnet10_backbone = dlib::avg_pool_everything<
    aresnet10_level1<
    aresnet10_level2<
    aresnet10_level3<
    aresnet_input<INPUT>>>>>;

using apredictor_t = dlib::loss_multiclass_log<dlib::fc<number_of_age_classes, aresnet10_backbone<dlib::input_rgb_image>>>;

//=====================================================Class===================================================
class FaceClassifier
{

private:

    anet_type net;
    agender_type gnet;
    apredictor_t anet;
    dlib::softmax<apredictor_t::subnet_type> asnet;

    uint8_t get_estimated_age(dlib::matrix<float, 1, number_of_age_classes>& p, float& confidence)
    {
        float estimated_age = (0.25f * p(0));
        confidence = p(0);

        for (uint16_t i = 1; i < number_of_age_classes; i++) {
            estimated_age += (static_cast<float>(i) * p(i));
            if (p(i) > confidence) confidence = p(i);
        }

        return std::lround(estimated_age);
    }

public:

    //------------------------------Constructor--------------------------------
    FaceClassifier(String ClassNetPath, String GenderNetPath, String AgeNetPath){

        dlib::deserialize(ClassNetPath)  >> net;
        dlib::deserialize(GenderNetPath) >> gnet;
        dlib::deserialize(AgeNetPath)    >> anet;

        // Usea Softmax for the last layer to estimate the age.
        asnet.subnet() = anet.subnet();
    }

    //-------------------Find Embeddings of a face/faces-----------------------
    dlib::matrix<float,0,1> FaceEmbeddings(Mat Face){
        std::vector<Mat> Faces;
        Faces.push_back(Face);
        std::vector<dlib::matrix<float,0,1>> Embeddings = FaceEmbeddings(Faces);
        return Embeddings[0];
    }

    std::vector<dlib::matrix<float,0,1>> FaceEmbeddings(std::vector<Mat> Faces){

        std::vector<dlib::matrix<dlib::rgb_pixel>> FacesDlib;

        for(int n=0; n<Faces.size(); ++n){
            Mat FaceCV;
            resize(Faces[n],FaceCV,Size(150,150));
            dlib::cv_image<dlib::bgr_pixel> FaceDlib(FaceCV);
            dlib::matrix<dlib::rgb_pixel> FaceMatrix;
            assign_image(FaceMatrix, FaceDlib);
            FacesDlib.push_back(FaceMatrix);
        }

        cout<<"Extracting facial embeddings... ";
        std::vector<dlib::matrix<float,0,1>> Output = net(FacesDlib);
        cout<<"Done!"<<endl;
        return Output;
    }

    //-------------------Estimate age of a face/faces-----------------------
    //warning... will probably be offensive
    int Age(Mat Face){
        resize(Face,Face,Size(64,64));

        dlib::cv_image<dlib::bgr_pixel> FaceDlib(Face);
        dlib::matrix<dlib::rgb_pixel> FaceMatrix;
        assign_image(FaceMatrix, FaceDlib);

        float confidence;
        dlib::matrix<float, 1, number_of_age_classes> p = mat(asnet(FaceMatrix));

        return get_estimated_age(p, confidence);
    }

    //-----------------Estimate gender of a face/faces---------------------
    //warning... will probably be offensive
    int Gender(Mat Face){
        resize(Face,Face,Size(32,32));

        dlib::cv_image<dlib::bgr_pixel> FaceDlib(Face);
        dlib::matrix<dlib::rgb_pixel> FaceMatrix;
        assign_image(FaceMatrix, FaceDlib);

        return gnet(FaceMatrix);
    }

};

#endif // FACECLASSIFIER_H












