//
//  Tree.cpp
//  myopencv
//
//  Created by lequan on 1/23/15.
//  Copyright (c) 2015 lequan. All rights reserved.
//

#include "Tree.h"
using namespace std;
using namespace cv;

inline float calculate_var(const vector<float>& v_1 ){
    if (v_1.size() == 0)
        return 0;
    Mat_<float> v1(v_1);
    float mean_1 = mean(v1)[0];
    float mean_2 = mean(v1.mul(v1))[0];
    return mean_2 - mean_1*mean_1;
    
}
inline float calculate_var(const Mat_<float>& v1){
    if (v1.rows==0)
    {
		return 0;
    }
	float mean_1 = mean(v1)[0];
    float mean_2 = mean(v1.mul(v1))[0];
    return mean_2 - mean_1*mean_1;
    
}

void Tree::Train(const vector<Mat_<uchar> >& images,
				 vector<int>& find_times,
				 const vector<int>& augmented_images,
                 const vector<Mat_<float> >& ground_truth_shapes,
				 const vector<int>& ground_truth_faces,
                 const vector<Mat_<float> >& current_shapes,
				 const vector<float>& current_fi,
				 const vector<float>& current_weight,
                 const vector<BoundingBox> & bounding_boxs,
                 const Mat_<float>& mean_shape,
                 const vector<Mat_<float> >& regression_targets,
                 const vector<int> index,
                 int stages,
                 int landmarkID
                 ){
    // set the parameter
    landmarkID_ = landmarkID; // start from 0
    max_numfeats_ = global_params.max_numfeats[stages];
    max_radio_radius_ = global_params.max_radio_radius[stages];
	max_probility_ = global_params.max_probility[stages];
    num_nodes_ = 1;
    num_leafnodes_ = 1;
    
    // index: indicate the training samples id in images
    int num_nodes_iter;
    int num_split;
	Mat_<float> shapes_residual((int)regression_targets.size(),2);
    // calculate regression targets: the difference between ground truth shapes and current shapes
    for(int i = 0;i < regression_targets.size();i++){
        shapes_residual(i,0) = regression_targets[i](landmarkID_,0);
        shapes_residual(i,1) = regression_targets[i](landmarkID_,1);
    }
    // initialize the root
    nodes_[0].issplit = false;
    nodes_[0].pnode = 0;
    nodes_[0].depth = 1;
    nodes_[0].cnodes[0] = 0;
    nodes_[0].cnodes[1] = 0;
    nodes_[0].isleafnode = 1;
    nodes_[0].thresh = 0;
    for (int i=0; i < 6;i++){
        nodes_[0].feat[i] = 1;
    }
	/*nodes_[0].landmarkId[0]=0;
	nodes_[0].landmarkId[1]=0;*/
    nodes_[0].ind_samples = index;
    nodes_[0].score=0;

    bool stop = 0;
    int num_nodes = 1;
    int num_leafnodes = 1;
    float thresh;
    float feat[6];
	//int markId[2];
    bool isvaild;
    vector<int> lcind,rcind;
    lcind.reserve(index.size());
    rcind.reserve(index.size());
    while(!stop){
        num_nodes_iter = num_nodes_;
        num_split = 0;
        for (int n = 0; n < num_nodes_iter; n++ ){
            if (!nodes_[n].issplit){
                if (nodes_[n].depth == max_depth_) {
                    if (nodes_[n].depth == 1){
                        nodes_[n].depth = 1;
                    }
                    nodes_[n].issplit = true;
                }
                else {
                    // separate the samples into left and right path
                    Splitnode(images,find_times,augmented_images,ground_truth_shapes,ground_truth_faces,current_shapes,current_fi,current_weight,bounding_boxs,mean_shape,shapes_residual,
						nodes_[n].ind_samples,thresh, feat,/*markId, */isvaild,lcind,rcind,stages);
                    // set the threshold and featture for current node
                    nodes_[n].feat[0] = feat[0];
                    nodes_[n].feat[1] = feat[1];
                    nodes_[n].feat[2] = feat[2];
                    nodes_[n].feat[3] = feat[3];
					nodes_[n].feat[4] = feat[4];
					nodes_[n].feat[5] = feat[5];
					/*nodes_[n].landmarkId[0] = markId[0];
					nodes_[n].landmarkId[1] = markId[1];*/
                    nodes_[n].thresh  = thresh;
                    nodes_[n].issplit = true;
                    nodes_[n].isleafnode = false;
                    nodes_[n].cnodes[0] = num_nodes ;
                    nodes_[n].cnodes[1] = num_nodes +1;
                    
                    //add left and right child nodes into the random tree
                    nodes_[num_nodes].ind_samples = lcind;
                    nodes_[num_nodes].issplit = false;
                    nodes_[num_nodes].pnode = n;
                    nodes_[num_nodes].depth = nodes_[n].depth + 1;
                    nodes_[num_nodes].cnodes[0] = 0;
                    nodes_[num_nodes].cnodes[1] = 0;
                    nodes_[num_nodes].isleafnode = true;

                    nodes_[num_nodes +1].ind_samples = rcind;
                    nodes_[num_nodes +1].issplit = false;
                    nodes_[num_nodes +1].pnode = n;
                    nodes_[num_nodes +1].depth = nodes_[n].depth + 1;
                    nodes_[num_nodes +1].cnodes[0] = 0;
                    nodes_[num_nodes +1].cnodes[1] = 0;
                    nodes_[num_nodes +1].isleafnode = true;
                    
                    num_split++;
                    num_leafnodes++;
                    num_nodes +=2;
                }
            }
            
            
        }
        if (num_split == 0){
            stop = 1;
        }
        else{
            num_nodes_ = num_nodes;
            num_leafnodes_ = num_leafnodes;
        }
    }
    
    id_leafnodes_.clear();
    /*for (int i=0;i < num_nodes_;i++){
        if (nodes_[i].isleafnode == 1){
            id_leafnodes_.push_back(i);
        }
    }*/
	for (int i=0;i < num_nodes_;i++){
		if (nodes_[i].isleafnode == 1){
			// compute leaf node's score
			float leafy_pos_weight=0;
			float leafy_neg_weight=0;
			for (int j=0;j<nodes_[i].ind_samples.size();++j)
			{
				if (find_times[augmented_images[nodes_[i].ind_samples[j]]]<=MAXFINDTIMES)
				{
					if (ground_truth_faces[nodes_[i].ind_samples[j]]==1)
					{
						leafy_pos_weight+=current_weight[nodes_[i].ind_samples[j]];
					}
					else
					{
						leafy_neg_weight+=current_weight[nodes_[i].ind_samples[j]];
					}
				}
			}
			//cout<<"leafy_pos_weight:"<<leafy_pos_weight<<" "<<"leafy_neg_weight:"<<leafy_neg_weight<<" ";
			nodes_[i].score=0.5*(((leafy_pos_weight-0.0)<FLT_EPSILON)?0:log(leafy_pos_weight))-0.5*(((leafy_neg_weight-0.0)<FLT_EPSILON)?0:log(leafy_neg_weight))/*/log(2.0)*/;
			//cout<<"score:"<<nodes_[i].score<<" "<<endl;
			// compute leaf node's score
			id_leafnodes_.push_back(i);
		}
	}
	/*if (id_leafnodes_.size()<8)
	{
		cout<<"leaf num:"<<id_leafnodes_.size()<<endl;
		system("pause");
	}*/
	
	//cout<<endl;
	//system("pause");
}
void Tree::Splitnode(const vector<Mat_<uchar> >& images,
					 vector<int>& find_times,
					 const vector<int>& augmented_images,
                     const vector<Mat_<float> >& ground_truth_shapes,
					 const vector<int>& ground_truth_faces,
                     const vector<Mat_<float> >& current_shapes,
					 const vector<float >& current_fi,
					 const vector<float >& current_weight,
                     const vector<BoundingBox> & bounding_box,
                     const Mat_<float>& mean_shape,
                     const Mat_<float>& shapes_residual,
                     const vector<int> &ind_samples_ori,
                     // output
                     float& thresh,
                     float* feat,
					 /*int* markId,*/
                     bool& isvaild,
                     vector<int>& lcind,
                     vector<int>& rcind,
					 int stage
                     ){
	vector<int> ind_samples;
	for (int i=0;i<ind_samples_ori.size();++i)
	{
		if(find_times[augmented_images[ind_samples_ori[i]]]>MAXFINDTIMES)
			continue;
		else
			ind_samples.push_back(ind_samples_ori[i]);
	}
    if (ind_samples.size() == 0){
        thresh = 0;
        //feat = new float[4];
        feat[0] = 0;
        feat[1] = 0;
        feat[2] = 0;
        feat[3] = 0;
		feat[4] = 0;
		feat[5] = 0;
        lcind.clear();
        rcind.clear();
        isvaild = 1;
        return;
    }
	
    // get candidate pixel locations
    RNG random_generator(getTickCount());
    Mat_<float> candidate_pixel_locations(max_numfeats_,6);
    for(unsigned int i = 0;i < max_numfeats_;i++){
        float x1 = random_generator.uniform(-1.0,1.0);
        float y1 = random_generator.uniform(-1.0,1.0);
        float x2 = random_generator.uniform(-1.0,1.0);
        float y2 = random_generator.uniform(-1.0,1.0);
        if((x1*x1 + y1*y1 > 1.0)||(x2*x2 + y2*y2 > 1.0)){
            i--;
            continue;
        }
       // cout << x1 << " "<<y1 <<" "<< x2<<" "<< y2<<endl;
        candidate_pixel_locations(i,0) = x1 * max_radio_radius_;
        candidate_pixel_locations(i,1) = y1 * max_radio_radius_;
        candidate_pixel_locations(i,2) = x2 * max_radio_radius_;
        candidate_pixel_locations(i,3) = y2 * max_radio_radius_;
		/*candidate_pixel_locations(i,4) =(int)random_generator.uniform(0.01,global_params.landmark_num-0.01);
		candidate_pixel_locations(i,5) =(int)random_generator.uniform(0.01,global_params.landmark_num-0.01);*/
		int tmp_idx=(int)random_generator.uniform(0.01,global_params.landmark_num-0.01);
		candidate_pixel_locations(i,4) =tmp_idx;
		candidate_pixel_locations(i,5) =tmp_idx;
    }
	// get landmark
	
	/*markId[0]=landmarkID_;
	markId[1]=landmarkID_;*/
    // get pixel difference feature
    Mat_<int> densities(max_numfeats_,(int)ind_samples.size());
	#pragma omp parallel for
    for (int i = 0;i < ind_samples.size();i++){
        Mat_<float> rotation;
        float scale;
        Mat_<float> temp = ProjectShape(current_shapes[ind_samples[i]],bounding_box[ind_samples[i]]);
        SimilarityTransform(temp,mean_shape,rotation,scale);
        // whether transpose or not ????
		
        for(int j = 0;j < max_numfeats_;j++)
		{
            float project_x1 = rotation(0,0) * candidate_pixel_locations(j,0) + rotation(0,1) * candidate_pixel_locations(j,1);
            float project_y1 = rotation(1,0) * candidate_pixel_locations(j,0) + rotation(1,1) * candidate_pixel_locations(j,1);
            project_x1 = scale * project_x1 * bounding_box[ind_samples[i]].width / 2.0;
            project_y1 = scale * project_y1 * bounding_box[ind_samples[i]].height / 2.0;
            int real_x1 = project_x1 + current_shapes[ind_samples[i]](candidate_pixel_locations(j,4),0);
            int real_y1 = project_y1 + current_shapes[ind_samples[i]](candidate_pixel_locations(j,4),1);
            real_x1 = max(0.0,min((double)real_x1,images[augmented_images[ind_samples[i]]].cols-1.0));
			real_y1 = max(0.0,min((double)real_y1,images[augmented_images[ind_samples[i]]].rows-1.0));
            
            float project_x2 = rotation(0,0) * candidate_pixel_locations(j,2) + rotation(0,1) * candidate_pixel_locations(j,3);
            float project_y2 = rotation(1,0) * candidate_pixel_locations(j,2) + rotation(1,1) * candidate_pixel_locations(j,3);
            project_x2 = scale * project_x2 * bounding_box[ind_samples[i]].width / 2.0;
            project_y2 = scale * project_y2 * bounding_box[ind_samples[i]].height / 2.0;
            int real_x2 = project_x2 + current_shapes[ind_samples[i]](candidate_pixel_locations(j,5),0);
            int real_y2 = project_y2 + current_shapes[ind_samples[i]](candidate_pixel_locations(j,5),1);
            real_x2 = max(0.0,min((double)real_x2,images[augmented_images[ind_samples[i]]].cols-1.0));
            real_y2 = max(0.0,min((double)real_y2,images[augmented_images[ind_samples[i]]].rows-1.0));
            
            densities(j,i) = ((int)(images[augmented_images[ind_samples[i]]](real_y1,real_x1))-(int)(images[augmented_images[ind_samples[i]]](real_y2,real_x2)));
        }
    }
    // pick the feature
    Mat_<int> densities_sorted = densities.clone();
    cv::sort(densities, densities_sorted, CV_SORT_ASCENDING);
	
	//separate shape samples
	vector<int> ind_samples_shape;
	for(int n=0;n<ind_samples.size();++n)
	{
		if (ground_truth_faces[ind_samples[n]]==1)
		{
			ind_samples_shape.push_back(ind_samples[n]);
		} 
	}
	Mat_<float> shapes_residual_shape(ind_samples_shape.size(),2);
	//#pragma omp parallel for
	for(int n=0,m=0;n<ind_samples.size();++n)
	{
		if (ground_truth_faces[ind_samples[n]]==1)
		{
			shapes_residual_shape(m,0)=shapes_residual(ind_samples[n],0);
			shapes_residual_shape(m,1)=shapes_residual(ind_samples[n],1);
			++m;
		} 
	}
	// threshold about shape
	float var_overall =(calculate_var(shapes_residual_shape.col(0))+calculate_var(shapes_residual_shape.col(1))) * ind_samples_shape.size();
	
	Mat_<float> cache_shape(max_numfeats_,2);
	Mat_<float> cache_face(max_numfeats_,2);
	#pragma omp parallel for
    for (int i = 0;i <max_numfeats_;i++){
		vector<float> lc1_shape,lc2_shape;
		vector<float> rc1_shape,rc2_shape;
		lc1_shape.reserve(ind_samples.size());
		lc2_shape.reserve(ind_samples.size());
		rc1_shape.reserve(ind_samples.size());
		rc2_shape.reserve(ind_samples.size());
		
		vector<float> lc_pos_weight,lc_neg_weight;
		vector<float> rc_pos_weight,rc_neg_weight;
		lc_pos_weight.reserve(ind_samples.size());
		lc_neg_weight.reserve(ind_samples.size());
		rc_pos_weight.reserve(ind_samples.size());
		rc_neg_weight.reserve(ind_samples.size());

		lc1_shape.clear();
		lc2_shape.clear();
		rc1_shape.clear();
		rc2_shape.clear();
		lc_pos_weight.clear();
		lc_neg_weight.clear();
		rc_pos_weight.clear();
		rc_neg_weight.clear();

		float total_lc_pos_weight=0,total_lc_neg_weight=0;
		float total_rc_pos_weight=0,total_rc_neg_weight=0;
		
		RNG random_generator2(getTickCount());
        int ind =(int)(ind_samples.size() * random_generator2.uniform(0.05,0.95));
        float threshold = densities_sorted(i,ind);
        for (int j=0;j < ind_samples.size();j++){
            if (densities(i,j) < threshold){
				if(ground_truth_faces[ind_samples[j]]==1)
				{
					lc1_shape.push_back(shapes_residual(ind_samples[j],0));
					lc2_shape.push_back(shapes_residual(ind_samples[j],1));
					
					lc_pos_weight.push_back(current_weight[ind_samples[j]]);
					total_lc_pos_weight+=current_weight[ind_samples[j]];
				}
				else
				{
					lc_neg_weight.push_back(current_weight[ind_samples[j]]);
					total_lc_neg_weight+=current_weight[ind_samples[j]];
				}
            }
            else{
				if(ground_truth_faces[ind_samples[j]]==1)
				{
					rc1_shape.push_back(shapes_residual(ind_samples[j],0));
					rc2_shape.push_back(shapes_residual(ind_samples[j],1)); 
					
					rc_pos_weight.push_back(current_weight[ind_samples[j]]);
					total_rc_pos_weight+=current_weight[ind_samples[j]];
				}
				else
				{
					rc_neg_weight.push_back(current_weight[ind_samples[j]]);
					total_rc_neg_weight+=current_weight[ind_samples[j]];
				}        
			}
        }
		// about shape
		float var_lc = (calculate_var(lc1_shape)+calculate_var(lc2_shape)) * lc1_shape.size();
		float var_rc = (calculate_var(rc1_shape)+calculate_var(rc2_shape)) * rc1_shape.size();
        float var_reduce = var_overall - var_lc - var_rc;
		cache_shape(i,0)=var_reduce;
		cache_shape(i,1)=threshold;

		// about face
		
		int total_sample_num = lc_pos_weight.size()+lc_neg_weight.size()+rc_pos_weight.size()+rc_neg_weight.size();
		int left_sample_num  = lc_pos_weight.size()+lc_neg_weight.size();
		int right_sample_num = rc_pos_weight.size()+rc_neg_weight.size(); 

		float total_weight = total_lc_pos_weight + total_lc_neg_weight + total_rc_pos_weight + total_rc_neg_weight;
		float total_lc_weight = total_lc_pos_weight + total_lc_neg_weight;
		float total_rc_weight = total_rc_pos_weight + total_rc_neg_weight;

		float entropy=0;
		float lc_entropy=0;
		float rc_entropy=0;

		if (total_sample_num==0)
		{
			lc_entropy=0;
			rc_entropy=0;
		}
		else
		{
			if (left_sample_num==0)
			{
				lc_entropy=0;
			}
			else
			{
				float entropy_tmp = total_lc_pos_weight / (total_lc_weight + FLT_MIN);
				//float entropy_tmp = lc_pos_weight.size()/(lc_pos_weight.size()+lc_neg_weight.size()+FLT_MIN);
				if ((entropy_tmp-0.0)<FLT_EPSILON)
				{
					lc_entropy=0;
				} 
				else
				{
					lc_entropy = -(total_lc_weight / (total_weight + FLT_MIN))*((entropy_tmp + FLT_MIN)*log(entropy_tmp + FLT_MIN) / log(2.0) + (1 - entropy_tmp + FLT_MIN)*log(1 - entropy_tmp + FLT_MIN) / log(2.0));
					//lc_entropy = -(left_sample_num/total_sample_num)*((entropy_tmp+FLT_MIN)*log(entropy_tmp+FLT_MIN)/log(2.0)+(1-entropy_tmp+FLT_MIN)*log(1-entropy_tmp+FLT_MIN)/log(2.0));
					//lc_entropy = -/*(left_sample_num/total_sample_num)**/((total_lc_pos_weight/*/(total_lc_pos_weight+total_lc_neg_weight+FLT_MIN)*/)*(entropy_tmp+FLT_MIN)*log(entropy_tmp+FLT_MIN)/log(2.0)+(total_lc_neg_weight/*/(total_lc_pos_weight+total_lc_neg_weight+FLT_MIN)*/)*(1-entropy_tmp+FLT_MIN)*log(1-entropy_tmp+FLT_MIN)/log(2.0));
				}

			}
			if (right_sample_num==0)
			{
				rc_entropy=0;
			}
			else
			{
				float entropy_tmp = total_rc_pos_weight/(total_rc_weight+FLT_MIN);
				//float entropy_tmp = rc_pos_weight.size()/(rc_pos_weight.size()+rc_neg_weight.size()+FLT_MIN);
				if ((entropy_tmp-0.0)<FLT_EPSILON)
				{
					rc_entropy=0;
				} 
				else
				{
					rc_entropy = -(total_rc_weight / (total_weight + FLT_MIN))*((entropy_tmp + FLT_MIN)*log(entropy_tmp + FLT_MIN) / log(2.0) + (1 - entropy_tmp + FLT_MIN)*log(1 - entropy_tmp + FLT_MIN) / log(2.0));
					//rc_entropy = -(right_sample_num/total_sample_num)*((entropy_tmp+FLT_MIN)*log(entropy_tmp+FLT_MIN)/log(2.0)+(1-entropy_tmp+FLT_MIN)*log(1-entropy_tmp+FLT_MIN)/log(2.0));
					//rc_entropy = -/*(right_sample_num/total_sample_num)**/((total_rc_pos_weight/*/(total_rc_pos_weight+total_rc_neg_weight+FLT_MIN)*/)*(entropy_tmp+FLT_MIN)*log(entropy_tmp+FLT_MIN)/log(2.0)+(total_rc_neg_weight/*/(total_rc_pos_weight+total_rc_neg_weight+FLT_MIN)*/)*(1-entropy_tmp+FLT_MIN)*log(1-entropy_tmp+FLT_MIN)/log(2.0));
				}
			}
		}
		entropy=lc_entropy+rc_entropy;
		cache_face(i,0)=entropy;
		cache_face(i,1)=threshold;
    }
	float thresh_shape=0;
	float thresh_face=0;

    float max_id_shape = 0;
	float max_id_face=0;

	float max_var_reductions = 0;
	float min_entropy=FLT_MAX;
	
	for (int i=0;i<cache_shape.rows;++i)
	{
		if (cache_shape(i,0) > max_var_reductions){
			max_var_reductions = cache_shape(i,0);
			thresh_shape = cache_shape(i,1);
			max_id_shape = i;
		}
	}
	for (int i=0;i<cache_face.rows;++i)
	{
		if (cache_face(i,0) < min_entropy){
			min_entropy = cache_face(i,0);
			thresh_face = cache_face(i,1);
			max_id_face = i;
		}
	}
	int max_id=0;
	if(random_generator.uniform(0.0,1.0)<max_probility_)
	{
		thresh=thresh_face;
		max_id=max_id_face;
	}
	else
	{
		thresh=thresh_shape;
		max_id=max_id_shape;
	}

    isvaild = 1;
    feat[0] =candidate_pixel_locations(max_id,0)/*/max_radio_radius_*/;
    feat[1] =candidate_pixel_locations(max_id,1)/*/max_radio_radius_*/;
    feat[2] =candidate_pixel_locations(max_id,2)/*/max_radio_radius_*/;
    feat[3] =candidate_pixel_locations(max_id,3)/*/max_radio_radius_*/;
	feat[4] =candidate_pixel_locations(max_id,4);
	feat[5] =candidate_pixel_locations(max_id,5);
//    cout << max_id<< " "<<max_var_reductions <<endl;
//    cout << feat[0] << " "<<feat[1] <<" "<< feat[2]<<" "<< feat[3]<<endl;
    lcind.clear();
    rcind.clear();
	
    for (int j=0;j < ind_samples.size();j++){
        if (densities(max_id,j) < thresh){
            lcind.push_back(ind_samples[j]);
        }
        else{
            rcind.push_back(ind_samples[j]);
        }
    }
}

void Tree::Write(std:: ofstream& fout){
    fout << landmarkID_<<endl;
    fout << max_depth_<<endl;
    fout << max_numnodes_<<endl;
    fout << num_leafnodes_<<endl;
    fout << num_nodes_<<endl;
    fout << max_numfeats_<<endl;
    fout << max_radio_radius_<<endl;
    fout << overlap_ration_ << endl;
	fout << max_probility_ << endl;
	fout << threshold << endl;
    
    fout << id_leafnodes_.size()<<endl;
    for (int i=0;i<id_leafnodes_.size();i++){
        fout << id_leafnodes_[i]<< " ";
    }
    fout <<endl;
    
    for (int i=0; i <max_numnodes_;i++){
        nodes_[i].Write(fout);
    }
}
void Tree::Read(std::ifstream& fin){
    fin >> landmarkID_;
    fin >> max_depth_;
    fin >> max_numnodes_;
    fin >> num_leafnodes_;
    fin >> num_nodes_;
    fin >> max_numfeats_;
    fin >> max_radio_radius_;
    fin >> overlap_ration_;
	fin >> max_probility_;
	fin >> threshold;
    int num ;
    fin >> num;
    id_leafnodes_.resize(num);
    for (int i=0;i<num;i++){
        fin >> id_leafnodes_[i];
    }
    
    for (int i=0; i <max_numnodes_;i++){
        nodes_[i].Read(fin);
    }
}


