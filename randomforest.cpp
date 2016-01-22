//
//  RandomForest.cpp
//  myopencv
//
//  Created by lequan on 1/24/15.
//  Copyright (c) 2015 lequan. All rights reserved.
//

#include "RandomForest.h"
#include "LBFRegressor.h"
using namespace std;
using namespace cv;

void GetCodefromRandomForestOnlyOnce(struct feature_node *binfeature,
                                           const int index,
                                           const vector<Tree>& rand_forest,
                                           const Mat_<uchar>& image,
                                           const Mat_<float>& shape,
                                           const BoundingBox& bounding_box,
                                           const Mat_<float>& rotation,
                                           const float scale){
    
    int leafnode_per_tree = pow(2.0,rand_forest[0].max_depth_-1);

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
void GlobalRegressionOnlyOnce(struct feature_node **binfeatures,
	Mat_<float>& current_shapes,
	BoundingBox& bounding_boxs,
	const Mat_<float>& mean_shape,
	vector<struct model*>& models
	){
		float tmp;
		float scale;
		Mat_<float>rotation;
		int num_residual = current_shapes.rows*2;
		Mat_<float> deltashape_bar(num_residual/2,2);
		Mat_<float> deltashape_bar1(num_residual/2,2);
		
#pragma omp parallel for
		for (int j=0;j<num_residual;j++){
			tmp = predict(models[j],binfeatures[0]);
			if (j < num_residual/2){
				deltashape_bar(j,0) = tmp;
			}
			else{
				deltashape_bar(j-num_residual/2,1) = tmp;
			}
		}

		SimilarityTransform(ProjectShape(current_shapes,bounding_boxs),mean_shape,rotation,scale);
		transpose(rotation,rotation);
		deltashape_bar1 = scale * deltashape_bar * rotation;
		current_shapes= ReProjectShape((ProjectShape(current_shapes,bounding_boxs)+deltashape_bar1),bounding_boxs);
		
}

int my_cmp(pair<float,int> p1, pair<float,int> p2)
{
	return p1.first < p2.first;
};

void RandomForest::Train(
                         const vector<Mat_<uchar>>& images,
						 const vector<int>& augmented_images,
                         vector<Mat_<float>>& ground_truth_shapes,
						 const vector<int>& ground_truth_faces,
                         vector<Mat_<float>>& current_shapes,
						 vector<float>& current_fi,
						 vector<float>& current_weight,
                         vector<BoundingBox> & bounding_boxs,
                         const Mat_<float>& mean_shape,
                         vector<Mat_<float> >& shapes_residual,
                         int stages,
                         vector<RandomForest>& RandomForest_){
    stages_ = stages;
	// all training samples
	int dbsize = (int)augmented_images.size();
	int Q = floor(dbsize*1.0/((1-overlap_ratio_)*max_numtrees_));
	int is,ie;
	vector<int> index;
	for (int k = 0; k<augmented_images.size();k++){
		index.push_back(k);
	}
	
    for (int i=0;i<num_landmark_;i++){
        clock_t tt = clock();
        for (int j =0;j <max_numtrees_; j++){
			// update weight
			#pragma omp parallel for
			for(int k=0;k<current_weight.size();++k)
			{
				current_weight[k]=exp(0.0-ground_truth_faces[k]*current_fi[k]);
			}
			// build RF
            rfs_[i][j].Train(images, augmented_images, ground_truth_shapes, ground_truth_faces , current_shapes, current_fi, current_weight, bounding_boxs, mean_shape,shapes_residual,index,stages_,i);

			// compute all samples's fi
			#pragma omp parallel for
			for (int n=0;n<augmented_images.size();++n)
			{				
				Mat_<float> rotation;
				float scale;
				SimilarityTransform(ProjectShape(current_shapes[n],bounding_boxs[n]),mean_shape,rotation,scale);
				int bincode=0;
				float score=0;
				GetResultfromTree(rfs_[i][j],images[augmented_images[n]],current_shapes[n],bounding_boxs[n],rotation,scale,&bincode,&score);
				current_fi[n] = current_fi[n] + score;
			}
			//float fiT=0,fiF=0;
			//float weightT=0,weigthF=0;
			////#pragma omp parallel for
			//for(int n=0;n<current_fi.size();++n)
			//{
			//	if (ground_truth_faces[n]==1)
			//	{
			//		fiT+=current_fi[n];
			//		weightT+=current_weight[n];
			//	} 
			//	else
			//	{
			//		fiF+=current_fi[n];
			//		weigthF+=current_weight[n];
			//	}
			//}
			//cout<<"fiTsum:"<<fiT<<" fiFsum:"<<fiF<<endl;
			//cout<<"weightTsum:"<<weightT<<" weightFsum:"<<weigthF<<endl;
			
			// remove false samples according to precision-recall 
			// sort fi with index
			vector<pair<float,int>> fiSort;
			fiSort.clear();
			for(int n=0;n<current_fi.size();++n)
			{
				fiSort.push_back(pair<float,int>(current_fi[n],n));
			}
			// ascent , small fi means false sample
			sort(fiSort.begin(),fiSort.end(),my_cmp);
			// compute recall
			// set threshold
			float max_recall=0,min_error=1;
			int idx_tmp=-1;

			vector<pair<float,float>> precise_recall;
			for (int n=1;n<fiSort.size();++n)
			{
				int true_pos=0;int false_neg=0;
				int true_neg=0;int false_pos=0;
				for(int m=0;m<fiSort.size();++m)
				{
					int isFace = ground_truth_faces[fiSort[m].second];
					// below the threshold as non-face
					if (m<n)
					{
						if (isFace==1)
						{
							false_neg++;
						} 
						else
						{
							true_neg++;
						}
					} 
					// up the threshold as face
					else
					{
						if (isFace==1)
						{
							true_pos++;
						} 
						else
						{
							false_pos++;
						}
					}
				}
				
				if (true_pos/(true_pos+false_neg+FLT_MIN)>=max_recall)
				{
					max_recall=true_pos/(true_pos+false_neg+FLT_MIN);
					precise_recall.push_back(pair<float,float>(true_pos/(true_pos+false_neg+FLT_MIN),false_pos/(false_pos+true_neg+FLT_MIN)));
					rfs_[i][j].threshold=fiSort[n].first;
				}
				else
					break;
			}
			
			cout<<"pre_recall in tree:"<<precise_recall[precise_recall.size()-1].first<<"  "<<precise_recall[precise_recall.size()-1].second<<endl;
			
			//remove false samples , perform hard sample
			int remove_num=0;
			//#pragma omp parallel for
			for(int n=0;n<current_fi.size();++n)
			{
				if (current_fi[n]<rfs_[i][j].threshold && ground_truth_faces[n]==-1)
				{
					int nSelected=0;
					while(1)
					{
						RNG random_generator(getTickCount());
						int tmp_idx=n;

						BoundingBox new_box;
						while (tmp_idx==n || ground_truth_faces[tmp_idx]==1)
						{
							tmp_idx = random_generator.uniform(0.0,ground_truth_faces.size()-1);
						}
						nSelected++;
						if (nSelected>=images[augmented_images[n]].rows*images[augmented_images[n]].cols*1.0)
						{
							Mat_<uchar> newImg = images[augmented_images[n]];
							if(random_generator.uniform(0.0,1.0)>0.8)
							{
								resize(images[augmented_images[tmp_idx]],newImg,newImg.size());
							}else
							{
								Point center = Point( images[augmented_images[n]].cols/2, images[augmented_images[n]].rows/2 );
								double angle = random_generator.uniform(0.0,360.0);
								double scale = 1.0;
								Mat_<float> rot_mat=getRotationMatrix2D(center,angle,scale);
								warpAffine( images[augmented_images[n]], newImg, rot_mat, newImg.size());
								rot_mat.release()；
							}// else read a new image, then resize it to images[augmented_images[n]]

							for (int i2=0;i2<images[augmented_images[n]].rows;++i2)
								for(int j2=0;j2<images[augmented_images[n]].cols;++j2)
									images[augmented_images[n]].data[i2*images[augmented_images[n]].cols+j2]=newImg.data[i2*images[augmented_images[n]].cols+j2];
							newImg.release();
							nSelected=0;
						}


						getRandomBox(images[augmented_images[n]],bounding_boxs[tmp_idx], new_box);
						Mat_<float> temp = ProjectShape(ground_truth_shapes[tmp_idx], bounding_boxs[tmp_idx]);
						ground_truth_shapes[n] = ReProjectShape(temp, new_box);
						
						temp = ProjectShape(current_shapes[tmp_idx], bounding_boxs[tmp_idx]);
						current_shapes[n]=ReProjectShape(temp, new_box);

						bounding_boxs[n]=new_box;
						
						bool tmp_isface=true;
						float tmp_fi=0;
						for (int s=0;s<=stages;++s)
						{
							int iRange=RandomForest_[s].rfs_.size();
							if (s==stages)
							{
								iRange=i;
							}
							for (int r=0;r<iRange;++r)
							{
								int jRange=RandomForest_[s].rfs_[i].size();
								if (r==i)
								{
									jRange=j;
								}
								for (int t=0;t<jRange;++t)
								{
									//get score 
									Mat_<float> rotation;
									float scale;
									SimilarityTransform(ProjectShape(current_shapes[n],bounding_boxs[n]),mean_shape,rotation,scale);
									int bincode=0;
									float score=0;
									GetResultfromTree(RandomForest_[s].rfs_[r][t],images[augmented_images[n]],current_shapes[n],bounding_boxs[n],rotation,scale,&bincode,&score);
									tmp_fi+=score;
									if (tmp_fi<RandomForest_[s].rfs_[r][t].threshold)
									{
										tmp_isface=false;
										break;
									}
								}
								if(!tmp_isface)break;
							}
							if(!tmp_isface)break;
							if ((s-1)>=0)
							{
								struct feature_node **binfeatures = new struct feature_node* [1];
								binfeatures[0] = new struct feature_node[RandomForest_[s-1].max_numtrees_*RandomForest_[s-1].num_landmark_+1];
								Mat_<float> rotation;
								float scale;
								SimilarityTransform(ProjectShape(current_shapes[n],bounding_boxs[n]),mean_shape,rotation,scale);
								for (int j =0; j <RandomForest_[s-1].num_landmark_; j++){
									GetCodefromRandomForestOnlyOnce(binfeatures[0], j*RandomForest_[s-1].max_numtrees_,RandomForest_[s-1].rfs_[j], images[augmented_images[n]], current_shapes[n],
										bounding_boxs[n], rotation, scale);
								}
								binfeatures[0][RandomForest_[s-1].num_landmark_ * RandomForest_[s-1].max_numtrees_].index = -1;
								binfeatures[0][RandomForest_[s-1].num_landmark_ * RandomForest_[s-1].max_numtrees_].value = -1;
								GlobalRegressionOnlyOnce(binfeatures, current_shapes[n],bounding_boxs[n],mean_shape,Models_[s-1]);
								delete[] binfeatures[0];
								delete[] binfeatures;
							}
						}
						if (tmp_isface)
						{
							current_fi[n]=tmp_fi;
							current_weight[n]=exp(0.0-ground_truth_faces[n]*current_fi[n]);
							break;
						}
					}
					remove_num++;
				} 
			}
			cout<<"remove "<<remove_num<<" samples\n";
		}
        float time = float(clock() - tt) / CLOCKS_PER_SEC;
        cout << "the train rf of "<< i <<"th landmark cost "<< time<<"s"<<endl;
		
    }

}
void RandomForest::Write(std::ofstream& fout){
    fout << stages_ <<endl;
    fout << max_numtrees_<<endl;
    fout << num_landmark_<<endl;
    fout << max_depth_ <<endl;
    fout << overlap_ratio_ <<endl;
    for (int i=0; i< num_landmark_;i++){
        for (int j = 0; j < max_numtrees_; j++){
            rfs_[i][j].Write(fout);
        }
    }
}
void RandomForest::Read(std::ifstream& fin){
    fin >> stages_;
    fin >> max_numtrees_;
    fin >> num_landmark_;
    fin >> max_depth_;
    fin >> overlap_ratio_;
    for (int i=0; i< num_landmark_;i++){
        for (int j = 0; j < max_numtrees_; j++){
            rfs_[i][j].Read(fin);
        }
    }
}

