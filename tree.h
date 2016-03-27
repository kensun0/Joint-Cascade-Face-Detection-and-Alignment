//
//  Tree.h
//  myopencv
//
//  Created by lequan on 1/23/15.
//  Copyright (c) 2015 lequan. All rights reserved.
//

#ifndef __myopencv__Tree__
#define __myopencv__Tree__

#include "LBF.h"

class Node {
public:
    //data
    bool issplit;
    int pnode;
    int depth;
    int cnodes[2];
    bool isleafnode;
    float thresh;
    float feat[6];
	//int landmarkId[2];
    std::vector<int> ind_samples;
    float score;

    //Constructors
    Node(){
        ind_samples.clear();
        issplit = 0;
        pnode = 0;
        depth = 0;
        cnodes[0] = 0;
        cnodes[1] = 0;
        isleafnode = 0;
        thresh = 0;
        feat[0] = 0;
        feat[1] = 0;
        feat[2] = 0;
        feat[3] = 0;
		feat[4] = 0;
		feat[5] = 0;
		/*landmarkId[0] = 0;
		landmarkId[1] = 0;*/
		score = 0;
    }
    void Write(std::ofstream& fout){
        fout << issplit<<" "<< pnode <<" "<<depth<<" " <<cnodes[0]<<" "<<cnodes[1]<<" "<<isleafnode<<" "
			<< thresh<<" "<<feat[0]<<" "<<feat[1]<<" "<<feat[2]<<" "<<feat[3]<<" "<<feat[4]<<" "<<feat[5]<<" "<<score<<std::endl;
		
		if(SAVEBIN&&issplit&&!isleafnode)
		{
			FILE* file;
			file=fopen("allNode","ab+");
			int tmp = (int)thresh;
			fwrite(&tmp,sizeof(int),1,file);
			fwrite(&feat[0],sizeof(float),1,file);
			fwrite(&feat[1],sizeof(float),1,file);
			fwrite(&feat[2],sizeof(float),1,file);
			fwrite(&feat[3],sizeof(float),1,file);
			fclose(file);
		}
		
    }
    void Read(std::ifstream& fin){
        fin >> issplit >> pnode >> depth >> cnodes[0] >> cnodes[1] >> isleafnode
        >> thresh >> feat[0] >> feat[1] >> feat[2] >> feat[3] >> feat[4] >> feat[5] >> score;
    }
};

class Tree{
public:
    
    // id of the landmark
    int landmarkID_;
    // depth of the tree:
    int max_depth_;
    // number of maximum nodes:
    int max_numnodes_;
    //number of leaf nodes and nodes
    int num_leafnodes_;
    int num_nodes_;
    
    // sample pixel featurs' number, use when training RF
    int max_numfeats_;
    float max_radio_radius_;
    float overlap_ration_;
   
	float max_probility_;
	float threshold;

    // leafnodes id
    std::vector<int> id_leafnodes_;
    // tree nodes
    std::vector<Node> nodes_;
    
    
    Tree(){
        overlap_ration_ = global_params.bagging_overlap;
        max_depth_ = global_params.max_depth;
        max_numnodes_ = pow(2.0, max_depth_)-1;
        nodes_.resize(max_numnodes_);

		max_probility_=1;
		threshold = 0;
    }
    void Train(const std::vector<cv::Mat_<uchar> >& images,
			   std::vector<int>& find_times,
		       const std::vector<int>& augmented_images,
               const std::vector<cv::Mat_<float> >& ground_truth_shapes,
			   const std::vector<int>& ground_truth_faces,
               const std::vector<cv::Mat_<float> >& current_shapes,
			   const std::vector<float>& current_fi,
			   const std::vector<float>& current_weight,
               const std::vector<BoundingBox> & bounding_boxs,
               const cv::Mat_<float>& mean_shape,
               const std::vector<cv::Mat_<float> >& regression_targets,
               const std::vector<int> index,
               int stages,
               int landmarkID
               );
    
    //Splite the node
    void Splitnode(const std::vector<cv::Mat_<uchar> >& images,
		           std::vector<int>& find_times,
				   const std::vector<int>& augmented_images,
                   const std::vector<cv::Mat_<float> >& ground_truth_shapes,
				   const std::vector<int>& ground_truth_faces,
                   const std::vector<cv::Mat_<float> >& current_shapes,
				   const std::vector<float>& current_fi,
				   const std::vector<float>& current_weight,
                   const std::vector<BoundingBox> & bounding_box,
                   const cv::Mat_<float>& mean_shape,
                   const cv::Mat_<float>& shapes_residual,
                   const std::vector<int> &ind_samples,
                   // output
                   float& thresh,
                   float* feat,
				   //int* markId,
                   bool& isvaild,
                   std::vector<int>& lcind,
                   std::vector<int>& rcind,
				   int stage
                   );
    
    //Predict
    void Predict();
    
    // Read/ write
    void Read(std::ifstream& fin);
    void Write(std:: ofstream& fout);
    
};





#endif /* defined(__myopencv__Tree__) */
