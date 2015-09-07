//
//  LBF.h
//  myopencv
//
//  Created by lequan on 1/24/15.
//  Copyright (c) 2015 lequan. All rights reserved.
//

#ifndef FACE_ALIGNMENT_H
#define FACE_ALIGNMENT_H
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <cctype>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include "cv.h"
#include <ctime>
#include <string>
#include <limits>
#include <algorithm>
#include <cmath>
#include <vector>
#include <fstream>
#include <numeric>   
#include <utility> 

struct Params{
    
    float bagging_overlap;
    int max_numtrees;
    int max_depth;
    int landmark_num;// to be decided
    int initial_num;
    
    int max_numstage;
    float max_radio_radius[10];
    int max_numfeats[10]; // number of pixel pairs
    int max_numthreshs;
	float max_probility[10];
};
extern Params global_params;
extern cv::string modelPath;
extern cv::string dataPath;
class BoundingBox{
    public:
        float start_x;
        float start_y;
        float width;
        float height;
        float centroid_x;
        float centroid_y;
        BoundingBox(){
            start_x = 0;
            start_y = 0;
            width = 0;
            height = 0;
            centroid_x = 0;
            centroid_y = 0;
        }; 
};
cv::Mat_<float> GetMeanShape(const std::vector<cv::Mat_<float> >& shapes,
                              const std::vector<BoundingBox>& bounding_box);
cv::Mat_<float> GetMeanShape2(const std::vector<cv::Mat_<float> >& shapes,
	const std::vector<BoundingBox>& bounding_box,const std::vector<int>& ground_truth_faces);

void GetShapeResidual(const std::vector<cv::Mat_<float> >& ground_truth_shapes,
                      const std::vector<cv::Mat_<float> >& current_shapes,
                      const std::vector<BoundingBox>& bounding_boxs,
                      const cv::Mat_<float>& mean_shape,
                      std::vector<cv::Mat_<float> >& shape_residuals);

cv::Mat_<float> ProjectShape(const cv::Mat_<float>& shape, const BoundingBox& bounding_box);
cv::Mat_<float> ReProjectShape(const cv::Mat_<float>& shape, const BoundingBox& bounding_box);
void SimilarityTransform(const cv::Mat_<float>& shape1, const cv::Mat_<float>& shape2,
                         cv::Mat_<float>& rotation,float& scale);
float calculate_covariance(const std::vector<float>& v_1,
                            const std::vector<float>& v_2);
void LoadData(std::string filepath,
              std::vector<cv::Mat_<uchar> >& images,
              std::vector<cv::Mat_<float> >& ground_truth_shapes,
              std::vector<BoundingBox> & bounding_box);
void LoadDataAdjust(std::string filepath,
              std::vector<cv::Mat_<uchar> >& images,
              std::vector<cv::Mat_<float> >& ground_truth_shapes,
              std::vector<BoundingBox> & bounding_box);
void LoadDataAdjust2(std::string filepath,
	std::vector<cv::Mat_<uchar> >& images,
	std::vector<cv::Mat_<float> >& ground_truth_shapes,
	std::vector<int>& ground_truth_faces,
	std::vector<BoundingBox> & bounding_box);
void LoadOpencvBbxData(std::string filepath,
                       std::vector<cv::Mat_<uchar> >& images,
                       std::vector<cv::Mat_<float> >& ground_truth_shapes,
                       std::vector<BoundingBox> & bounding_boxs
                       );
void LoadCofwTrainData(std::vector<cv::Mat_<uchar> >& images,
                       std::vector<cv::Mat_<float> >& ground_truth_shapes,
                       std::vector<BoundingBox>& bounding_boxs);
void LoadCofwTestData(std::vector<cv::Mat_<uchar> >& images,
                       std::vector<cv::Mat_<float> >& ground_truth_shapes,
                       std::vector<BoundingBox>& bounding_boxs);

BoundingBox CalculateBoundingBox(cv::Mat_<float>& shape);
BoundingBox CalculateBoundingBox2(cv::Mat_<float>& shape);
cv::Mat_<float> LoadGroundTruthShape(std::string& filename);
void adjustImage(cv::Mat_<uchar>& img,
                 cv::Mat_<float>& ground_truth_shape,
                 BoundingBox& bounding_box);
void adjustImage2(cv::Mat_<uchar>& img,
	cv::Mat_<float>& ground_truth_shape,
	BoundingBox& bounding_box);

void  TrainModel(std::vector<std::string> trainDataName);
float TestModel(std::vector<std::string> testDataName);
int FaceDetectionAndAlignment(const char* inputname);
void ReadGlobalParamFromFile(cv::string path);
float CalculateError(const cv::Mat_<float>& ground_truth_shape, const cv::Mat_<float>& predicted_shape);


void getRandomBox(const cv::Mat_<uchar>& image, const BoundingBox& old_box, BoundingBox& new_box);

#define SAVEBIN (0)

#define MAXHEIGHT (120)

#define MAXHEIGHT2 (240)

#define ONLYSHAPE (1)

#define MINHEIGHT (10)

#endif
