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
#include "cv.hpp"
#include <ctime>
#include <string>
#include <limits>
#include <algorithm>
#include <cmath>
#include <vector>
#include <fstream>
#include <numeric>   
#include <utility> 
#include <omp.h>
using namespace std;
using namespace cv;

#define SAVEBIN (0)
#define MAXHEIGHT_POS (128)
#define MAXHEIGHT_NEG (320)
#define MINHEIGHT (16)
#define CROP (0.1)
#define FRAC (0.0)
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
extern std::string modelPath;
extern std::string dataPath;
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

void InitializeGlobalParam();

Mat_<float> GetMeanShape(const vector<Mat_<float> >& shapes,
                              const vector<BoundingBox>& bounding_box);
Mat_<float> GetMeanShape2(const vector<Mat_<float> >& shapes,
	const vector<BoundingBox>& bounding_box,const vector<int>& ground_truth_faces);

void GetShapeResidual(const vector<Mat_<float> >& ground_truth_shapes,
                      const vector<Mat_<float> >& current_shapes,
                      const vector<BoundingBox>& bounding_boxs,
                      const Mat_<float>& mean_shape,
                      vector<Mat_<float> >& shape_residuals);
void GetShapeResidual2(vector<int> shape_index,
	const vector<Mat_<float> >& ground_truth_shapes,
	const vector<Mat_<float> >& current_shapes,
	const vector<BoundingBox>& bounding_boxs,
	const Mat_<float>& mean_shape,
	vector<Mat_<float> >& shape_residuals);

cv::Mat_<float> ProjectShape(const cv::Mat_<float>& shape, const BoundingBox& bounding_box);
cv::Mat_<float> ReProjectShape(const cv::Mat_<float>& shape, const BoundingBox& bounding_box);
void SimilarityTransform(const cv::Mat_<float>& shape1, const cv::Mat_<float>& shape2,
                         cv::Mat_<float>& rotation,float& scale);
float calculate_covariance(const std::vector<float>& v_1,
                            const std::vector<float>& v_2);
float CalculateError(const cv::Mat_<float>& ground_truth_shape, const cv::Mat_<float>& predicted_shape);

BoundingBox CalculateBoundingBox(Mat_<uchar>& img, cv::Mat_<float>& shape);
void adjustImage(Mat_<uchar>& img, Mat_<float>& ground_truth_shape, BoundingBox& bounding_box);
void getRandomBox(const cv::Mat_<uchar>& image, const BoundingBox& old_box, BoundingBox& new_box);
void cropBoundingBox(Mat_<uchar>& img, BoundingBox box, BoundingBox& newbox/*, Mat_<float> shapeMat_, <float>& newshape*/);


#endif
