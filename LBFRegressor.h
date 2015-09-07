//
//  LBFRegressor.h
//  myopencv
//
//  Created by lequan on 1/24/15.
//  Copyright (c) 2015 lequan. All rights reserved.
//

#ifndef __myopencv__LBFRegressor__
#define __myopencv__LBFRegressor__

#include "RandomForest.h"
#include "liblinear/linear.h"
class LBFRegressor{
public:
    std::vector<RandomForest> RandomForest_;
    std::vector<std::vector<struct model*> > Models_;
    cv::Mat_<float> mean_shape_;
    std::vector<cv::Mat_<float> > shapes_residual_;
    int max_numstage_;
public:
    LBFRegressor(){
        max_numstage_ = global_params.max_numstage;
        RandomForest_.resize(max_numstage_);
        Models_.resize(max_numstage_);
    }
    ~LBFRegressor(){
        
    }
    void Read(std::ifstream& fin);
    void Write(std::ofstream& fout);
    void Load(std::string path);
    void Save(std::string path);
    struct feature_node ** DeriveBinaryFeat(const RandomForest& randf,
                                            const std::vector<cv::Mat_<uchar> >& images,
											const std::vector<int>& augmented_images,
                                            const std::vector<cv::Mat_<float> >& current_shapes,
                                            const std::vector<BoundingBox> & bounding_boxs);
	struct feature_node ** DeriveBinaryFeat3(const RandomForest& randf,
		const std::vector<cv::Mat_<uchar> >& images,
		const std::vector<int>& augmented_images,
		const std::vector<cv::Mat_<float> >& current_shapes,
		const std::vector<int >& ground_truth_faces,
		const std::vector<BoundingBox> & bounding_boxs);
	struct feature_node ** DeriveBinaryFeat2(const RandomForest& randf,
		const std::vector<cv::Mat_<uchar> >& images,
		const std::vector<int>& augmented_images,
		const std::vector<cv::Mat_<float> >& current_shapes,
		const std::vector<BoundingBox> & bounding_boxs,std::vector<bool> & result_face,float& score,int& fcount,bool& fface);
    void ReleaseFeatureSpace(struct feature_node ** binfeatures,
                             int num_train_sample);
    int   GetCodefromTree(const Tree& tree,
                          const cv::Mat_<uchar>& image,
                          const cv::Mat_<float>& shapes,
                          const BoundingBox& bounding_box,
                          const cv::Mat_<float>& rotation,
                          const float scale);
    void GetCodefromRandomForest(struct feature_node *binfeature,
                                 const int index,
                                 const cv::vector<Tree>& rand_forest,
                                 const cv::Mat_<uchar>& image,
                                 const cv::Mat_<float>& shape,
                                 const BoundingBox& bounding_box,
                                 const cv::Mat_<float>& rotation,
                                 const float scale);
	bool GetCodefromRandomForest2(struct feature_node *binfeature,
		const int index,
		const cv::vector<Tree>& rand_forest,
		const cv::Mat_<uchar>& image,
		const cv::Mat_<float>& shape,
		const BoundingBox& bounding_box,
		const cv::Mat_<float>& rotation,
		const float scale,float& score,int& fcount,bool& fface);
    void GlobalRegression(struct feature_node **binfeatures,
                          const std::vector<cv::Mat_<float> >& shapes_residual,
                          std::vector<cv::Mat_<float> >& current_shapes,
                          const std::vector<BoundingBox> & bounding_boxs,
                          const cv::Mat_<float>& mean_shape,
                          //Mat_<float>& W,
                          std::vector<struct model*>& models,
                          int num_feature,
                          int num_train_sample,
                          int stage);
	void GlobalRegression2(struct feature_node **binfeatures,
		const std::vector<cv::Mat_<float> >& shapes_residual,
		std::vector<cv::Mat_<float> >& current_shapes,
		const std::vector<BoundingBox> & bounding_boxs,
		const cv::Mat_<float>& mean_shape,
		//Mat_<float>& W,
		std::vector<struct model*>& models,
		int num_feature,
		const std::vector<int> & augmented_images,
		const std::vector<int> & ground_truth_faces,
		int stage);
    
    void GlobalPrediction(struct feature_node**,
                          std::vector<cv::Mat_<float> >& current_shapes,
                          const std::vector<BoundingBox> & bounding_boxs,
                          int stage);
    
    void Train(const std::vector<cv::Mat_<uchar> >& images,
               const std::vector<cv::Mat_<float> >& ground_truth_shapes,
			   const std::vector<int>& ground_truth_faces,
               const std::vector<BoundingBox> & bounding_boxs);
    
    std::vector<cv::Mat_<float> > Predict(const std::vector<cv::Mat_<uchar> >& images,
                                           const std::vector<BoundingBox>& bounding_boxs,
                                           const std::vector<cv::Mat_<float> >& ground_truth_shapes,
										   int initial_num,std::vector<bool>& result_face);
    cv::Mat_<float>  Predict(const cv::Mat_<uchar>& image,
                              const BoundingBox& bounding_box,
                              int initial_num, bool& isface,int& fcount);
    void WriteGlobalParam(std::ofstream& fout);
    void ReadGlobalParam(std::ifstream& fin);
    void WriteRegressor(std::ofstream& fout);
    void ReadRegressor(std::ifstream& fin);
    

};

void   GetResultfromTree(const Tree& tree,
	const cv::Mat_<uchar>& image,
	const cv::Mat_<float>& shapes,
	const BoundingBox& bounding_box,
	const cv::Mat_<float>& rotation,
	const float scale,int* bincode,float* score);

#endif
