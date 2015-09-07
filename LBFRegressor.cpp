//
//  LBFRegressor.cpp
//  myopencv
//
//  Created by lequan on 1/24/15.
//  Copyright (c) 2015 lequan. All rights reserved.
//
#include "HighClock.h"
#include "LBFRegressor.h"
using namespace std;
using namespace cv;


struct feature_node ** LBFRegressor::DeriveBinaryFeat(
	const RandomForest& randf,
	const vector<Mat_<uchar> >& images,
	const vector<int>& augmented_images,
	const vector<Mat_<float> >& current_shapes,
	const vector<BoundingBox> & bounding_boxs){

		// initilaize the memory for binfeatures
		struct feature_node **binfeatures;
		binfeatures = new struct feature_node* [augmented_images.size()];
		for (int i=0;i<augmented_images.size();i++){
			binfeatures[i] = new struct feature_node[randf.max_numtrees_*randf.num_landmark_+1];
		}

		//    int bincode;
		//    int ind;
		//    int leafnode_per_tree = pow(2.0,(randf.max_depth_-1));

		Mat_<float> rotation;
		float scale;

		// extract feature for each samples
		// #pragma omp parallel for

		for (int i=0;i < augmented_images.size();i++){
			SimilarityTransform(ProjectShape(current_shapes[i],bounding_boxs[i]),mean_shape_,rotation,scale);
			//cout<<rotation<<endl;
			//#pragma omp parallel for
			
			for (int j =0; j <randf.num_landmark_; j++){
				GetCodefromRandomForest(binfeatures[i], j*randf.max_numtrees_,randf.rfs_[j], images[augmented_images[i]], current_shapes[i],
					bounding_boxs[i], rotation, scale);
				//            for(int k = 0; k< randf.max_numtrees_;k++){
				//                bincode = GetCodefromTree(randf.rfs_[j][k],images[i],current_shapes[i],bounding_boxs[i],rotation,scale);
				//                ind = j * randf.max_numtrees_ + k;
				//                binfeatures[i][ind].index = leafnode_per_tree * ind + bincode;
				//                binfeatures[i][ind].value = 1;
				//            }
				
			}
			binfeatures[i][randf.num_landmark_ * randf.max_numtrees_].index = -1;
			binfeatures[i][randf.num_landmark_ * randf.max_numtrees_].value = -1;
		}
		return binfeatures;
}

struct feature_node ** LBFRegressor::DeriveBinaryFeat3(
	const RandomForest& randf,
	const vector<Mat_<uchar> >& images,
	const vector<int>& augmented_images,
	const vector<Mat_<float> >& current_shapes,
	const vector<int>& ground_truth_faces,
	const vector<BoundingBox> & bounding_boxs){

		vector<int> index;
		for (int i=0;i<augmented_images.size();++i)
		{
			if (ground_truth_faces[i]==1)
			{
				index.push_back(i);
			}
		}
		
		
		// initilaize the memory for binfeatures
		struct feature_node **binfeatures;
		binfeatures = new struct feature_node* [index.size()];
		for (int i=0;i<index.size();i++){
			binfeatures[i] = new struct feature_node[randf.max_numtrees_*randf.num_landmark_+1];
		}

		//    int bincode;
		//    int ind;
		//    int leafnode_per_tree = pow(2.0,(randf.max_depth_-1));

		Mat_<float> rotation;
		float scale;

		// extract feature for each samples
		// #pragma omp parallel for

		for (int i=0;i < index.size();i++){
			SimilarityTransform(ProjectShape(current_shapes[index[i]],bounding_boxs[index[i]]),mean_shape_,rotation,scale);
			//cout<<rotation<<endl;
			//#pragma omp parallel for

			for (int j =0; j <randf.num_landmark_; j++){
				GetCodefromRandomForest(binfeatures[i], j*randf.max_numtrees_,randf.rfs_[j], images[augmented_images[index[i]]], current_shapes[index[i]],
					bounding_boxs[index[i]], rotation, scale);
				//            for(int k = 0; k< randf.max_numtrees_;k++){
				//                bincode = GetCodefromTree(randf.rfs_[j][k],images[i],current_shapes[i],bounding_boxs[i],rotation,scale);
				//                ind = j * randf.max_numtrees_ + k;
				//                binfeatures[i][ind].index = leafnode_per_tree * ind + bincode;
				//                binfeatures[i][ind].value = 1;
				//            }

			}
			binfeatures[i][randf.num_landmark_ * randf.max_numtrees_].index = -1;
			binfeatures[i][randf.num_landmark_ * randf.max_numtrees_].value = -1;
		}
		return binfeatures;
}

struct feature_node ** LBFRegressor::DeriveBinaryFeat2(
                                    const RandomForest& randf,
                                    const vector<Mat_<uchar>>& images,
									const vector<int>& augmented_images,
                                    const vector<Mat_<float>>& current_shapes,
                                    const vector<BoundingBox>& bounding_boxs, vector<bool>& result_face, float& score,int& fcount,bool& fface){
    
    // initilaize the memory for binfeatures
    struct feature_node **binfeatures;
    binfeatures = new struct feature_node* [augmented_images.size()];
    for (int i=0;i<augmented_images.size();i++){
         binfeatures[i] = new struct feature_node[randf.max_numtrees_*randf.num_landmark_+1];
    }
    
//    int bincode;
//    int ind;
//    int leafnode_per_tree = pow(2.0,(randf.max_depth_-1));
    
    Mat_<float> rotation;
    float scale;

    // extract feature for each samples
   // #pragma omp parallel for
	
    for (int i=0;i < augmented_images.size();i++){
        SimilarityTransform(ProjectShape(current_shapes[i],bounding_boxs[i]),mean_shape_,rotation,scale);
		//cout<<rotation<<endl;
       	//#pragma omp parallel for
		bool isface=true;
        for (int j =0; j <randf.num_landmark_; j++){
	       bool tmp=GetCodefromRandomForest2(binfeatures[i], j*randf.max_numtrees_,randf.rfs_[j], images[augmented_images[i]], current_shapes[i],
                                    bounding_boxs[i], rotation, scale,score,fcount,fface);
//            for(int k = 0; k< randf.max_numtrees_;k++){
//                bincode = GetCodefromTree(randf.rfs_[j][k],images[i],current_shapes[i],bounding_boxs[i],rotation,scale);
//                ind = j * randf.max_numtrees_ + k;
//                binfeatures[i][ind].index = leafnode_per_tree * ind + bincode;
//                binfeatures[i][ind].value = 1;
//            }
#if ONLYSHAPE
           if (!tmp)
           {
			   isface=false;
			   /*break;*/
           } 
#endif
        }
		result_face.push_back(isface);
        binfeatures[i][randf.num_landmark_ * randf.max_numtrees_].index = -1;
        binfeatures[i][randf.num_landmark_ * randf.max_numtrees_].value = -1;
    }
    return binfeatures;
}

// get code of one landmark.
// index: the start index of tree.
void LBFRegressor::GetCodefromRandomForest(struct feature_node *binfeature,
                                           const int index,
                                           const vector<Tree>& rand_forest,
                                           const Mat_<uchar>& image,
                                           const Mat_<float>& shape,
                                           const BoundingBox& bounding_box,
                                           const Mat_<float>& rotation,
                                           const float scale){
    
    int leafnode_per_tree = pow(2.0,rand_forest[0].max_depth_-1);
    /*float landmark_x = shape(rand_forest[0].landmarkID_,0);
    float landmark_y = shape(rand_forest[0].landmarkID_,1);*/

    for (int iter = 0;iter<rand_forest.size();iter++){
        int currnode = 0;
        int bincode = 1;
        for(int i = 0;i<rand_forest[iter].max_depth_-1;i++){
            float x1 = rand_forest[iter].nodes_[currnode].feat[0];
            float y1 = rand_forest[iter].nodes_[currnode].feat[1];
            float x2 = rand_forest[iter].nodes_[currnode].feat[2];
            float y2 = rand_forest[iter].nodes_[currnode].feat[3];

			int mark1 = rand_forest[iter].nodes_[currnode].feat[4];
			int mark2 = rand_forest[iter].nodes_[currnode].feat[5];
            
			float landmark_x1 = shape(mark1,0);
			float landmark_y1 = shape(mark1,1);

			float landmark_x2 = shape(mark2,0);
			float landmark_y2 = shape(mark2,1);

            float project_x1 = rotation(0,0) * x1 + rotation(0,1) * y1;
            float project_y1 = rotation(1,0) * x1 + rotation(1,1) * y1;
            project_x1 = scale * project_x1 * bounding_box.width / 2.0;
            project_y1 = scale * project_y1 * bounding_box.height / 2.0;
            int real_x1 = (int)(project_x1 + landmark_x1);
            int real_y1 = (int)(project_y1 + landmark_y1);
            real_x1 = max(0,min(real_x1,image.cols-1));
            real_y1 = max(0,min(real_y1,image.rows-1));
            
            float project_x2 = rotation(0,0) * x2 + rotation(0,1) * y2;
            float project_y2 = rotation(1,0) * x2 + rotation(1,1) * y2;
            project_x2 = scale * project_x2 * bounding_box.width / 2.0;
            project_y2 = scale * project_y2 * bounding_box.height / 2.0;
            int real_x2 = (int)(project_x2 + landmark_x2);
            int real_y2 = (int)(project_y2 + landmark_y2);
            real_x2 = max(0,min(real_x2,image.cols-1));
            real_y2 = max(0,min(real_y2,image.rows-1));
            
            int pdf = (int)(image(real_y1,real_x1))-(int)(image(real_y2,real_x2));
            if (pdf < rand_forest[iter].nodes_[currnode].thresh){
                currnode =rand_forest[iter].nodes_[currnode].cnodes[0];
            }
            else{
                currnode =rand_forest[iter].nodes_[currnode].cnodes[1];
                bincode += pow(2.0, rand_forest[iter].max_depth_-2-i);
            }
        }
        binfeature[index+iter].index = leafnode_per_tree*(index+iter)+bincode;
        binfeature[index+iter].value = 1;
        
    }
}

bool LBFRegressor::GetCodefromRandomForest2(struct feature_node *binfeature,
	const int index,
	const vector<Tree>& rand_forest,
	const Mat_<uchar>& image,
	const Mat_<float>& shape,
	const BoundingBox& bounding_box,
	const Mat_<float>& rotation,
	const float scale,float& score,int& fcount,bool& fface){

		int leafnode_per_tree = pow(2.0,rand_forest[0].max_depth_-1);
		float landmark_x = shape(rand_forest[0].landmarkID_,0);
		float landmark_y = shape(rand_forest[0].landmarkID_,1);
		bool isface=true;
		
		for (int iter = 0;iter<rand_forest.size();iter++)
		{
			int currnode = 0;
			int bincode = 1;
			if (fface)
			{
				fcount++;
			}
			for(int i = 0;i<rand_forest[iter].max_depth_-1;i++)
			{
				float x1 = rand_forest[iter].nodes_[currnode].feat[0];
				float y1 = rand_forest[iter].nodes_[currnode].feat[1];
				float x2 = rand_forest[iter].nodes_[currnode].feat[2];
				float y2 = rand_forest[iter].nodes_[currnode].feat[3];

				int mark1 = rand_forest[iter].nodes_[currnode].feat[4];
				int mark2 = rand_forest[iter].nodes_[currnode].feat[5];

				float landmark_x1 = shape(mark1,0);
				float landmark_y1 = shape(mark1,1);

				float landmark_x2 = shape(mark2,0);
				float landmark_y2 = shape(mark2,1);

				float project_x1 = rotation(0,0) * x1 + rotation(0,1) * y1;
				float project_y1 = rotation(1,0) * x1 + rotation(1,1) * y1;
				project_x1 = scale * project_x1 * bounding_box.width / 2.0;
				project_y1 = scale * project_y1 * bounding_box.height / 2.0;
				int real_x1 = (int)(project_x1 + landmark_x1);
				int real_y1 = (int)(project_y1 + landmark_y1);
				real_x1 = max(0,min(real_x1,image.cols-1));
				real_y1 = max(0,min(real_y1,image.rows-1));

				float project_x2 = rotation(0,0) * x2 + rotation(0,1) * y2;
				float project_y2 = rotation(1,0) * x2 + rotation(1,1) * y2;
				project_x2 = scale * project_x2 * bounding_box.width / 2.0;
				project_y2 = scale * project_y2 * bounding_box.height / 2.0;
				int real_x2 = (int)(project_x2 + landmark_x2);
				int real_y2 = (int)(project_y2 + landmark_y2);
				real_x2 = max(0,min(real_x2,image.cols-1));
				real_y2 = max(0,min(real_y2,image.rows-1));

				int pdf = (int)(image(real_y1,real_x1))-(int)(image(real_y2,real_x2));
				if (pdf < rand_forest[iter].nodes_[currnode].thresh){
					currnode =rand_forest[iter].nodes_[currnode].cnodes[0];
				}
				else{
					currnode =rand_forest[iter].nodes_[currnode].cnodes[1];
					bincode += pow(2.0, rand_forest[iter].max_depth_-2-i);
				}
			}
				//
			
			
#if ONLYSHAPE
			score+=rand_forest[iter].nodes_[currnode].score;
			if (score<rand_forest[iter].threshold)
			{
				isface=false;
				fface=false;
				//continue;
			}
#endif
#if ONLYSHAPE
			/*if (!isface)
			{
				continue;
			}*/
#endif
			binfeature[index+iter].index = leafnode_per_tree*(index+iter)+bincode;
			binfeature[index+iter].value = 1;
		}
		return isface;
}


int  LBFRegressor::GetCodefromTree(const Tree& tree,
                                   const Mat_<uchar>& image,
                                   const Mat_<float>& shape,
                                   const BoundingBox& bounding_box,
                                   const Mat_<float>& rotation,
                                   const float scale){
    int currnode = 0;
    int bincode = 1;
    for(int i = 0;i<tree.max_depth_-1;i++){
        float x1 = tree.nodes_[currnode].feat[0];
        float y1 = tree.nodes_[currnode].feat[1];
        float x2 = tree.nodes_[currnode].feat[2];
        float y2 = tree.nodes_[currnode].feat[3];
        
		int mark1 = tree.nodes_[currnode].feat[4];
		int mark2 = tree.nodes_[currnode].feat[5];

		/*float landmark_x1 = shape(mark1,0);
		float landmark_y1 = shape(mark1,1);

		float landmark_x2 = shape(mark2,0);
		float landmark_y2 = shape(mark2,1);*/

        float project_x1 = rotation(0,0) * x1 + rotation(0,1) * y1;
        float project_y1 = rotation(1,0) * x1 + rotation(1,1) * y1;
        project_x1 = scale * project_x1 * bounding_box.width / 2.0;
        project_y1 = scale * project_y1 * bounding_box.height / 2.0;
        int real_x1 = project_x1 + shape(mark1,0);
        int real_y1 = project_y1 + shape(mark1,1);
        real_x1 = max(0,min(real_x1,image.cols-1));
        real_y1 = max(0,min(real_y1,image.rows-1));
        
        float project_x2 = rotation(0,0) * x2 + rotation(0,1) * y2;
        float project_y2 = rotation(1,0) * x2 + rotation(1,1) * y2;
        project_x2 = scale * project_x2 * bounding_box.width / 2.0;
        project_y2 = scale * project_y2 * bounding_box.height / 2.0;
        int real_x2 = project_x2 + shape(mark2,0);
        int real_y2 = project_y2 + shape(mark2,1);
        real_x2 = max(0,min(real_x2,image.cols-1));
        real_y2 = max(0,min(real_y2,image.rows-1));
        
        int pdf = (int)(image(real_y1,real_x1))-(int)(image(real_y2,real_x2));
        if (pdf < tree.nodes_[currnode].thresh){
            currnode =tree.nodes_[currnode].cnodes[0];
        }
        else{
            currnode =tree.nodes_[currnode].cnodes[1];
            bincode += pow(2.0, tree.max_depth_-2-i);
        }
    }
    return bincode;
};

void  GetResultfromTree(const Tree& tree,
	const Mat_<uchar>& image,
	const Mat_<float>& shape,
	const BoundingBox& bounding_box,
	const Mat_<float>& rotation,
	const float scale,int* bincode,float* score){
		int currnode = 0;
		*bincode = 1;
		for(int i = 0;i<tree.max_depth_-1;i++){
			float x1 = tree.nodes_[currnode].feat[0];
			float y1 = tree.nodes_[currnode].feat[1];
			float x2 = tree.nodes_[currnode].feat[2];
			float y2 = tree.nodes_[currnode].feat[3];

			int mark1 = tree.nodes_[currnode].feat[4];
			int mark2 = tree.nodes_[currnode].feat[5];

			/*float landmark_x1 = shape(mark1,0);
			float landmark_y1 = shape(mark1,1);

			float landmark_x2 = shape(mark2,0);
			float landmark_y2 = shape(mark2,1);*/

			float project_x1 = rotation(0,0) * x1 + rotation(0,1) * y1;
			float project_y1 = rotation(1,0) * x1 + rotation(1,1) * y1;
			project_x1 = scale * project_x1 * bounding_box.width / 2.0;
			project_y1 = scale * project_y1 * bounding_box.height / 2.0;
			int real_x1 = project_x1 + shape(mark1,0);
			int real_y1 = project_y1 + shape(mark1,1);
			real_x1 = max(0,min(real_x1,image.cols-1));
			real_y1 = max(0,min(real_y1,image.rows-1));

			float project_x2 = rotation(0,0) * x2 + rotation(0,1) * y2;
			float project_y2 = rotation(1,0) * x2 + rotation(1,1) * y2;
			project_x2 = scale * project_x2 * bounding_box.width / 2.0;
			project_y2 = scale * project_y2 * bounding_box.height / 2.0;
			int real_x2 = project_x2 + shape(mark2,0);
			int real_y2 = project_y2 + shape(mark2,1);
			real_x2 = max(0,min(real_x2,image.cols-1));
			real_y2 = max(0,min(real_y2,image.rows-1));

			int pdf = (int)(image(real_y1,real_x1))-(int)(image(real_y2,real_x2));
			if (pdf < tree.nodes_[currnode].thresh){
				currnode =tree.nodes_[currnode].cnodes[0];
			}
			else{
				currnode =tree.nodes_[currnode].cnodes[1];
				*bincode += pow(2.0, tree.max_depth_-2-i);
			}
		}
		*score=tree.nodes_[currnode].score;

};

void LBFRegressor::GlobalRegression(struct feature_node **binfeatures,
                                    const vector<Mat_<float> >& shapes_residual,
                                    vector<Mat_<float> >& current_shapes,
                                    const vector<BoundingBox> & bounding_boxs,
                                    const Mat_<float>& mean_shape,
                                    //Mat_<float>& W,
                                    vector<struct model*>& models,
                                    int num_feature,
                                    int num_train_sample,
                                    int stage
                                    ){
    // shapes_residual: n*(l*2)
    // construct the problem(expect y)
    struct problem* prob = new struct problem;
    prob -> l = num_train_sample;
    prob -> n = num_feature;
    prob -> x = binfeatures;
    prob -> bias = -1;
    
    // construct the parameter
    struct parameter* param = new struct parameter;
    param-> solver_type = L2R_L2LOSS_SVR_DUAL;
  //  param-> solver_type = L2R_L2LOSS_SVR;
    param->C = 1.0/num_train_sample;
    param->p = 0;
    param->eps = 0.00001;
    //param->eps = 0.001;
    
    // initialize the y
    int num_residual = shapes_residual[0].rows*2;
    float** yy = new float*[num_residual];
    
    for (int i=0;i<num_residual;i++){
        yy[i] = new float[num_train_sample];
    }
    for (int i=0;i < num_train_sample;i++){
        for (int j=0;j<num_residual;j++){
            if (j < num_residual/2){
                yy[j][i] = shapes_residual[i](j,0);
            }
            else{
                yy[j][i] = shapes_residual[i](j-num_residual/2,1);
            }
        }
    }
    
    //train
    models.clear();
    models.resize(num_residual);
    #pragma omp parallel for
    for (int i=0;i < num_residual;i++){
        clock_t t1 = clock();
        cout << "Train "<< i <<"th landmark"<<endl;
        prob->y = yy[i];
        check_parameter(prob, param);
        struct model* lbfmodel  = train(prob, param);
        models[i] = lbfmodel;
        float time =float(clock() - t1) / CLOCKS_PER_SEC;
        cout << "linear regression of one landmark cost "<< time <<"s"<<endl;
    }
    // update the current shape and shapes_residual
    float tmp;
    float scale;
    Mat_<float>rotation;
    Mat_<float> deltashape_bar(num_residual/2,2);
    Mat_<float> deltashape_bar1(num_residual/2,2);
    for (int i=0;i<num_train_sample;i++){
        #pragma omp parallel for
        for (int j=0;j<num_residual;j++){
            tmp = predict(models[j],binfeatures[i]);
            if (j < num_residual/2){
                deltashape_bar(j,0) = tmp;
            }
            else{
                deltashape_bar(j-num_residual/2,1) = tmp;
            }
        }
        // transfer or not to be decided
        // now transfer
        SimilarityTransform(ProjectShape(current_shapes[i],bounding_boxs[i]),mean_shape,rotation,scale);
        transpose(rotation,rotation);
        deltashape_bar1 = scale * deltashape_bar * rotation;
        current_shapes[i] = ReProjectShape((ProjectShape(current_shapes[i],bounding_boxs[i])+deltashape_bar1),bounding_boxs[i]);
        
        //updata shapes_residual
       // shapes_residual[i] = shapes_residual[i] - deltashape_bar;
    }
}

void LBFRegressor::GlobalRegression2(struct feature_node **binfeatures,
	const vector<Mat_<float> >& shapes_residual,
	vector<Mat_<float> >& current_shapes,
	const vector<BoundingBox> & bounding_boxs,
	const Mat_<float>& mean_shape,
	//Mat_<float>& W,
	vector<struct model*>& models,
	int num_feature,
	const vector<int>& augmented_images,
	const vector<int>& ground_truth_faces,
	int stage
	){
		vector<int> index;
		for (int i=0;i<augmented_images.size();++i)
		{
			if (ground_truth_faces[i]==1)
			{
				index.push_back(i);
			}
		}
		
		// shapes_residual: n*(l*2)
		// construct the problem(expect y)
		struct problem* prob = new struct problem;
		prob -> l = index.size();
		prob -> n = num_feature;
		prob -> x = binfeatures;
		prob -> bias = -1;

		// construct the parameter
		struct parameter* param = new struct parameter;
		param-> solver_type = L2R_L2LOSS_SVR_DUAL;
		//  param-> solver_type = L2R_L2LOSS_SVR;
		param->C = 1.0/index.size();
		param->p = 0;
		param->eps = 0.00001;
		//param->eps = 0.001;

		// initialize the y
		int num_residual = shapes_residual[index[0]].rows*2;
		float** yy = new float*[num_residual];

		for (int i=0;i<num_residual;i++){
			yy[i] = new float[index.size()];
		}
		for (int i=0;i < index.size();i++){
			for (int j=0;j<num_residual;j++){
				if (j < num_residual/2){
					yy[j][i] = shapes_residual[index[i]](j,0);
				}
				else{
					yy[j][i] = shapes_residual[index[i]](j-num_residual/2,1);
				}
			}
		}

		//train
		models.clear();
		models.resize(num_residual);
#pragma omp parallel for
		for (int i=0;i < num_residual;i++){
			clock_t t1 = clock();
			cout << "Train "<< i <<"th landmark"<<endl;
			prob->y = yy[i];
			check_parameter(prob, param);
			struct model* lbfmodel  = train(prob, param);
			models[i] = lbfmodel;
			float time =float(clock() - t1) / CLOCKS_PER_SEC;
			cout << "linear regression of one landmark cost "<< time <<"s"<<endl;
		}
		// update the current shape and shapes_residual
		float tmp;
		float scale;
		Mat_<float>rotation;
		Mat_<float> deltashape_bar(num_residual/2,2);
		Mat_<float> deltashape_bar1(num_residual/2,2);
		for (int i=0;i<index.size();i++){
#pragma omp parallel for
			for (int j=0;j<num_residual;j++){
				tmp = predict(models[j],binfeatures[i]);
				if (j < num_residual/2){
					deltashape_bar(j,0) = tmp;
				}
				else{
					deltashape_bar(j-num_residual/2,1) = tmp;
				}
			}
			// transfer or not to be decided
			// now transfer
			SimilarityTransform(ProjectShape(current_shapes[index[i]],bounding_boxs[index[i]]),mean_shape,rotation,scale);
			transpose(rotation,rotation);
			deltashape_bar1 = scale * deltashape_bar * rotation;
			current_shapes[index[i]] = ReProjectShape((ProjectShape(current_shapes[index[i]],bounding_boxs[index[i]])+deltashape_bar1),bounding_boxs[index[i]]);

			//updata shapes_residual
			// shapes_residual[i] = shapes_residual[i] - deltashape_bar;
		}
}

void LBFRegressor::GlobalPrediction(struct feature_node** binfeatures,
                                    vector<Mat_<float> >& current_shapes,
                                    const vector<BoundingBox> & bounding_boxs,
                                    int stage){
    int num_train_sample = (int)current_shapes.size();
    int num_residual = current_shapes[0].rows*2;
    float tmp;
    float scale;
    Mat_<float>rotation;
    Mat_<float> deltashape_bar(num_residual/2,2);
   // #pragma omp parallel for
    for (int i=0;i<num_train_sample;i++){
		//cout<<current_shapes[i]<<endl;
        current_shapes[i] = ProjectShape(current_shapes[i],bounding_boxs[i]);
		//cout<<current_shapes[i]<<endl;
        float t =(float)cvGetTickCount();
       	//#pragma omp parallel for
        for (int j=0;j<num_residual;j++){
            tmp = predict(Models_[stage][j],binfeatures[i]);
            if (j < num_residual/2){
                deltashape_bar(j,0) = tmp;
            }
            else{
                deltashape_bar(j-num_residual/2,1) = tmp;
            }
        }
		//cout<<deltashape_bar<<endl;
        // transfer or not to be decided
        // now transfer
        SimilarityTransform(current_shapes[i],mean_shape_,rotation,scale);
        transpose(rotation,rotation);
        deltashape_bar = scale * deltashape_bar * rotation;
        current_shapes[i] = ReProjectShape((current_shapes[i]+deltashape_bar),bounding_boxs[i]);
    }
}

void LBFRegressor::Train(const vector<Mat_<uchar> >& images,
                         const vector<Mat_<float> >& ground_truth_shapes,
						 const vector<int>& ground_truth_faces,
                         const vector<BoundingBox> & bounding_boxs){
    
    // data augmentation and multiple initialization
    
    vector<int> augmented_images;  // use index instead of  real img
    vector<BoundingBox> augmented_bounding_boxs;
    vector<Mat_<float> > augmented_ground_truth_shapes;
	vector<int> augmented_ground_truth_faces;
    vector<Mat_<float> > current_shapes;
	vector<float> current_fi;
	vector<float> current_weight;
    
    RNG random_generator(getTickCount());
    for(int i = 0;i < images.size();i++){
        for(int j = 0;j < global_params.initial_num;j++){
            int index = 0;
            do{
                // index = (i+j+1) % (images.size());
                index = random_generator.uniform(0, (int)images.size());
            }while(index == i);
            
			if (ground_truth_faces[i]==-1)
			{
				for (int k=0;k<10;++k)
				{
					int tmp_idx;
					do{
						// index = (i+j+1) % (images.size());
						tmp_idx = random_generator.uniform(0, (int)images.size());
					}while(tmp_idx == i);
					
					BoundingBox new_box;
					getRandomBox(images[i],bounding_boxs[tmp_idx], new_box);
					Mat_<float> temp1 = ProjectShape(ground_truth_shapes[tmp_idx], bounding_boxs[tmp_idx]);
					//ground_truth_shapes[n] = ReProjectShape(temp, new_box);

					//Mat_<float> temp2 = ProjectShape(current_shapes[tmp_idx], bounding_boxs[tmp_idx]);
					//current_shapes[n]=ReProjectShape(temp, new_box);
					//bounding_boxs[n]=new_box;

					// 1. Select ground truth shapes of other images as initial shapes
					augmented_images.push_back(i);
					augmented_ground_truth_shapes.push_back(ReProjectShape(temp1, new_box).clone());
					augmented_ground_truth_faces.push_back(ground_truth_faces[i]);
					augmented_bounding_boxs.push_back(new_box);

					// 2. Project current shape to bounding box of ground truth shapes
					//Mat_<float> temp = ProjectShape(ground_truth_shapes[index], bounding_boxs[index]);
					//temp = ReProjectShape(temp, bounding_boxs[i]);
					current_shapes.push_back(ReProjectShape(temp1, new_box).clone());
					current_fi.push_back(0);
					current_weight.push_back(1);
				}
			}
			else
			{
				// 1. Select ground truth shapes of other images as initial shapes
				augmented_images.push_back(i);
				augmented_ground_truth_shapes.push_back(ground_truth_shapes[i].clone());
				augmented_ground_truth_faces.push_back(ground_truth_faces[i]);
				augmented_bounding_boxs.push_back(bounding_boxs[i]);

				// 2. Project current shape to bounding box of ground truth shapes
				Mat_<float> temp = ProjectShape(ground_truth_shapes[index], bounding_boxs[index]);
				temp = ReProjectShape(temp, bounding_boxs[i]);
				current_shapes.push_back(temp);
				current_fi.push_back(0);
				current_weight.push_back(1);
			}

           
        }
    }
	vector<int> shape_index;
	for (int i=0;i<augmented_images.size();++i)
	{
		if (augmented_ground_truth_faces[i]==1)
		{
			shape_index.push_back(i);
		}
	}
    // get mean shape from training shapes(only origin train images)
    mean_shape_ = GetMeanShape2(ground_truth_shapes,bounding_boxs,ground_truth_faces);
    cout << mean_shape_<<endl;
    // train random forest
    int num_feature = global_params.landmark_num * global_params.max_numtrees * pow(2.0,(global_params.max_depth-1));
    int num_train_sample = (int)augmented_images.size();

    float t0 =(float)cvGetTickCount();
    for (int stage = 0; stage < global_params.max_numstage; stage++){
        float t1 =(float)cvGetTickCount();
        GetShapeResidual(augmented_ground_truth_shapes,current_shapes,augmented_bounding_boxs,
                         mean_shape_,shapes_residual_);
        
        cout << "train random forest of "<< stage <<" stage" <<endl;
        RandomForest_[stage].Train(images, augmented_images, augmented_ground_truth_shapes, augmented_ground_truth_faces,current_shapes, current_fi,current_weight,augmented_bounding_boxs, mean_shape_, shapes_residual_, stage, RandomForest_);
        float t2 = (float)cvGetTickCount();
        cout << "the random forest of "<< stage<<" stage has been trained, cost "<< (t2-t1)/((float)cvGetTickFrequency()*1000*1000) <<" s"<<endl<<endl;

        cout << "derive binary codes given learned random forest in stage"<< stage << endl;
        struct feature_node ** binfeatures ;
		//vector<bool> isface;isface.push_back(true);
        binfeatures = DeriveBinaryFeat3(RandomForest_[stage], images, augmented_images, current_shapes,augmented_ground_truth_faces, augmented_bounding_boxs);
        float t3 = (float)cvGetTickCount();
        cout << "derive binary features of "<< stage<<" stage has been trained, cost "<< (t3-t2)/((float)cvGetTickFrequency()*1000*1000) <<" s"<<endl<<endl;
        
        cout << "learn global linear regression given binary feature" << endl;
        GlobalRegression2(binfeatures, shapes_residual_, current_shapes, augmented_bounding_boxs, mean_shape_, Models_[stage], num_feature, augmented_images,augmented_ground_truth_faces, stage);
		ReleaseFeatureSpace(binfeatures,(int)shape_index.size());
        
        //calculate the error
       /* float MRSE_sum = 0;
		int tmp_num=0;
        for (int i =0; i<current_shapes.size();i++){
			if (augmented_ground_truth_faces[i]==1)
			{
				MRSE_sum += CalculateError(augmented_ground_truth_shapes[i], current_shapes[i]);
				tmp_num++;
			}
        }
        cout <<"stage "<<stage<<", error: "<<MRSE_sum/tmp_num<<endl;*/
        
        //calculate the remaining time
        float t4 = (float)cvGetTickCount();
        cout << "the linear model of "<< stage<<" stage has been trained, cost "<< (t4-t3)/((float)cvGetTickFrequency()*1000*1000) <<" s"<<endl<<endl;

        cout << "the "<<stage<<" has completed, cost "<<(t4-t0)/((float)cvGetTickFrequency()*1000*1000) <<" s"<<endl;
        cout << "Remaining time is about "<< (t4-t0)/((float)cvGetTickFrequency()*1000*1000*(stage+1))*(global_params.max_numstage-stage-1)<< "s"<<endl<<endl;
    }
}
void LBFRegressor::ReleaseFeatureSpace(struct feature_node ** binfeatures,
                         int num_train_sample){
    for (int i = 0;i < num_train_sample;i++){
            delete[] binfeatures[i];
    }
    delete[] binfeatures;
}

vector<Mat_<float> > LBFRegressor::Predict(const vector<Mat_<uchar> >& images,
                                    const vector<BoundingBox>& bounding_boxs,
                                    const vector<Mat_<float> >& ground_truth_shapes,
                                    int initial_num,vector<bool>& result_face){
    vector<int> image_index;
	for (int i=0;i<images.size();++i)
	{
		image_index.push_back(i);
	}
	
    vector<Mat_<float> > current_shapes;
	//HighClock* timer1=new HighClock();timer1->Start();
    for (int i=0; i<images.size();i++){
        Mat_<float> current_shape = ReProjectShape(mean_shape_, bounding_boxs[i]);
        current_shapes.push_back(current_shape);
		//cout<<current_shape<<endl;
    }
    float MRSE_sum = 0;
    for (int i =0; i<current_shapes.size();i++){
        MRSE_sum += CalculateError(ground_truth_shapes[i], current_shapes[i]);
    }
    cout <<"mean shape "<<", error: "<<MRSE_sum/current_shapes.size()<<endl;
	//FILE* testfile;testfile=fopen("binfeature","wb");
	//HighClock* timer1=new HighClock();/*timer1->Start();*/float totaltime=0;
	float score=0;
	
    int stage1 =0;
    for ( int stage = 0; stage < global_params.max_numstage; stage++){
        if(stage<global_params.max_numstage){
            stage1 = stage;
        }
        else{
            stage1 = global_params.max_numstage-1;
        }
		//timer1->Start();
        struct feature_node ** binfeatures ;
		//HighClock* timer2=new HighClock();timer2->Start();
        //binfeatures = DeriveBinaryFeat2(RandomForest_[stage1],images,image_index,current_shapes,bounding_boxs,result_face,score);
		//timer2->Stop();cout<<"total time2:"<<timer2->GetTime()<<endl;
		//fwrite(binfeatures);
		//HighClock* timer3=new HighClock();timer3->Start();
        GlobalPrediction(binfeatures, current_shapes,bounding_boxs,stage1);
		//cout<<current_shapes[0]<<endl;
		//timer3->Stop();cout<<"total time3:"<<timer3->GetTime()<<endl;
		//timer1->Stop();totaltime+=timer1->GetTime();
		ReleaseFeatureSpace(binfeatures,images.size());
		
		//HighClock* timer4=new HighClock();timer4->Start();
        /*float MRSE_sum = 0;
        for (int i =0; i<current_shapes.size();i++){
            MRSE_sum += CalculateError(ground_truth_shapes[i], current_shapes[i]);
        }
        cout <<"stage "<<stage<<", error: "<<MRSE_sum/current_shapes.size()<<endl;*/
		//timer4->Stop();cout<<"total time4:"<<timer4->GetTime()<<endl;
    }
    //timer1->Stop();cout<<"total time:"<<totaltime/current_shapes.size()<<endl;
    return current_shapes;
}

Mat_<float>  LBFRegressor::Predict(const cv::Mat_<uchar>& image,
                                    const BoundingBox& bounding_box,
									int initial_num,bool& isface,int& fcount){
    vector<Mat_<uchar> > images;
	vector<int> image_index;
	image_index.push_back(0);
    vector<Mat_<float> > current_shapes;
    vector<BoundingBox>  bounding_boxs;


    images.push_back(image);
    bounding_boxs.push_back(bounding_box);
    current_shapes.push_back(ReProjectShape(mean_shape_, bounding_box));
   
	float score=0;
//    Mat img = imread("/Users/lequan/workspace/LBF/Datasets/lfpw/testset/image_0078.png");
//    // draw result :: red
//    for(int j = 0;j < global_params.landmark_num;j++){
//        circle(img,Point2d(current_shapes[0](j,0),current_shapes[0](j,1)),1,Scalar(255,255,255),-1,8,0);
//    }
//    imshow("result", img);
//    waitKey(0);
//    string name = "example mean.jpg";
//    imwrite(name,img);
    bool fface=true;
    for ( int stage = 0; stage < global_params.max_numstage; stage++){
		struct feature_node ** binfeatures ;
		vector<bool> face;
		face.clear();
		double t1 = (double)cvGetTickCount();
		binfeatures = DeriveBinaryFeat2(RandomForest_[stage],images,image_index,current_shapes,bounding_boxs,face,score,fcount,fface);
		double t2 = (double)cvGetTickCount();
		//cout << "the DeriveBinaryFeat2 cost "<< (t2-t1)/((double)cvGetTickFrequency()) <<" us"<<endl<<endl;
#if ONLYSHAPE
		if (!face[0])
		{
			isface=false;
			//break;
		}
#endif
        double t11 = (double)cvGetTickCount();
		GlobalPrediction(binfeatures, current_shapes,bounding_boxs,stage);
		double t21 = (double)cvGetTickCount();
		//cout << "the GlobalPrediction cost "<< (t21-t11)/((double)cvGetTickFrequency()) <<" us"<<endl<<endl;
        ReleaseFeatureSpace(binfeatures, images.size());
        
//        Mat image = imread("/Users/lequan/workspace/LBF/Datasets/afw/image_0078.png");
//        // draw result :: red
//        for(int j = 0;j < global_params.landmark_num;j++){
//            circle(image,Point2d(current_shapes[0](j,0),current_shapes[0](j,1)),1,Scalar(255,255,255),-1,8,0);
//        }
//        imshow("result", image);
//        waitKey(0);
//        string name = "example "+ to_string(stage) + ".jpg";
//        imwrite(name,image);

    }
    return current_shapes[0];
}

void LBFRegressor::Save(string path){
    cout << endl<<"Saving model..." << endl;
    ofstream fout;
	locale::global(locale(""));
    fout.open(path);
	locale::global(locale("C"));
    // write the Regressor to file
    WriteGlobalParam(fout);
    WriteRegressor(fout);
    fout.close();
    cout << "End" << endl;

    
}

void LBFRegressor::Load(string path){
    cout << "Loading model from "<< path  << endl;
    ifstream fin;
	locale::global(locale(""));
    fin.open(path);
	locale::global(locale("C"));
    ReadGlobalParam(fin);
    ReadRegressor(fin);
    fin.close();
    cout << "End"<<endl;
}
void  LBFRegressor::WriteGlobalParam(ofstream& fout){
    fout << global_params.bagging_overlap << endl;
    fout << global_params.max_numtrees << endl;
    fout << global_params.max_depth << endl;
    fout << global_params.max_numthreshs << endl;
    fout << global_params.landmark_num << endl;
    fout << global_params.initial_num << endl;
    fout << global_params.max_numstage << endl;
    
    for (int i = 0; i< global_params.max_numstage; i++){
        fout << global_params.max_radio_radius[i] << " ";
        
    }
    fout << endl;
    
    for (int i = 0; i < global_params.max_numstage; i++){
        fout << global_params.max_numfeats[i] << " ";
    }
    fout << endl;
}
void  LBFRegressor::WriteRegressor(ofstream& fout){
    for(int i = 0;i < global_params.landmark_num;i++){
        fout << mean_shape_(i,0)<<" "<< mean_shape_(i,1)<<" ";
		if(SAVEBIN)
		{
			FILE* file;
			file=fopen("meanShape","ab+");
			fwrite(&mean_shape_(i,0),sizeof(float),1,file);
			fwrite(&mean_shape_(i,1),sizeof(float),1,file);
			fclose(file);
		}
    }
	
	
    fout<<endl;
    ofstream fout_reg;
	locale::global(locale(""));
    fout_reg.open(modelPath + "\\Regressor6.model",ios::binary);
	locale::global(locale("C"));
    for (int i=0; i < global_params.max_numstage; i++ ){
        RandomForest_[i].Write(fout);
        fout << Models_[i].size()<< endl;
        for (int j=0; j<Models_[i].size();j++){
            save_model_bin(fout_reg, Models_[i][j]);
        }
    }
    fout_reg.close();
}
void  LBFRegressor::ReadGlobalParam(ifstream& fin){
    fin >> global_params.bagging_overlap;
    fin >> global_params.max_numtrees;
    fin >> global_params.max_depth;
    fin >> global_params.max_numthreshs;
    fin >> global_params.landmark_num;
    fin >> global_params.initial_num;
    fin >> global_params.max_numstage;
    
    for (int i = 0; i< global_params.max_numstage; i++){
        fin >> global_params.max_radio_radius[i];
    }
    
    for (int i = 0; i < global_params.max_numstage; i++){
        fin >> global_params.max_numfeats[i];
    }
}

void LBFRegressor::ReadRegressor(ifstream& fin){
    mean_shape_ = Mat::zeros(global_params.landmark_num,2,CV_32FC1);
    for(int i = 0;i < global_params.landmark_num;i++){
        fin >> mean_shape_(i,0) >> mean_shape_(i,1);
    }
    ifstream fin_reg;
	locale::global(locale(""));
    fin_reg.open(modelPath + "\\Regressor6.model",ios::binary);
	locale::global(locale("C"));
	//cout<<"111:"<<global_params.max_numstage<<endl;
    for (int i=0; i < global_params.max_numstage; i++ ){
        RandomForest_[i].Read(fin);
        int num =0;
        fin >> num;
		//cout<<"i:"<<num<<endl;
        Models_[i].resize(num);
        for (int j=0;j<num;j++){
            Models_[i][j]   = load_model_bin(fin_reg);
        }
    }
    fin_reg.close();
}

