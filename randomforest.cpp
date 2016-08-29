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
#define max(a,b) (((a) > (b)) ? (a) : (b))
#define min(a,b) (((a) < (b)) ? (a) : (b))
int perfomtimes = 0;
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

void RandomForest::Train(vector<Mat_<uchar> >& images,
	vector<int>& augmented_images,
	vector<Mat_<float> >& ground_truth_shapes,
	vector<int>& ground_truth_faces,
	vector<Mat_<float> >& current_shapes,
	vector<float>& current_fi,
	vector<double>& current_weight,
	vector<BoundingBox> & bounding_boxs,
	const Mat_<float>& mean_shape,
	vector<Mat_<float> >& shapes_residual,
	int stages,
	vector<RandomForest>& RandomForest_,
	vector<vector<struct model*> > Models_, int posLenth) {
	stages_ = stages;
	// all training samples
	vector<int> index;
	for (int k = 0; k < augmented_images.size(); k++)
		index.push_back(k);

	for (int i = 0; i < num_landmark_; i++)
	{
		clock_t tt_clk = clock();
		for (int j = 0; j < max_numtrees_; j++)
		{
			clock_t ss_clk = clock();
			// update weight
			double totalWeight = 0;
			for (int k = 0; k < current_weight.size(); ++k) {
				current_weight[k] = min(100, exp(0.0 - ground_truth_faces[k] * current_fi[k]));
				totalWeight += current_weight[k];
			}
			for (int k = 0; k < current_weight.size(); ++k)
			{
				current_weight[k] /= totalWeight;
			}
				
			// build RF
			rfs_[i][j].Train(images, augmented_images, ground_truth_shapes, ground_truth_faces, current_shapes, current_fi, current_weight, bounding_boxs, mean_shape, shapes_residual, index, stages_, i);

			// compute all samples's fi
#pragma omp parallel for
			for (int n = 0; n < augmented_images.size(); ++n)
			{
				Mat_<float> rotation;
				float scale;
				SimilarityTransform(ProjectShape(current_shapes[n], bounding_boxs[n]), mean_shape, rotation, scale);
				int bincode = 0;
				float score = 0;
				GetResultfromTree(rfs_[i][j], images[augmented_images[n]], current_shapes[n], bounding_boxs[n], rotation, scale, &bincode, &score);
				current_fi[n] = current_fi[n] + score;
			}
			float fiT = 0, fiF = 0;
			double weightT = 0, weigthF = 0;
			int numT = 0, numF = 0;
			double maxF = DBL_MIN, minF = DBL_MAX;
			double maxW = DBL_MIN, minW = DBL_MAX;
			int maxImgIdx = 0;
			for (int n = 0; n < current_fi.size(); ++n)
			{
				if (ground_truth_faces[n] == 1) {
					fiT += current_fi[n];
					weightT += current_weight[n];
					++numT;
				}
				else {
					fiF += current_fi[n];
					weigthF += current_weight[n];
					++numF;
				}
				if (current_fi[n] > maxF) {
					maxF = current_fi[n];
				}
				if (current_fi[n] < minF) {
					minF = current_fi[n];
				}
				if (current_weight[n] > maxW) {
					maxW = current_weight[n];
					maxImgIdx = n;
				}
				if (current_weight[n] < minW) {
					minW = current_weight[n];
				}
			}
			cout << "fiTsum:" << fiT << " fiFsum:" << fiF << endl;
			cout << "weightTsum:" << weightT << " weightFsum:" << weigthF << endl;
			cout << "max     fi:" << maxF << "\t\tmin     fi:" << minF << endl;
			cout << "max weight:" << maxW << "\t\tmin weight:" << minW << endl;
			cout << "avg     fiT:" << fiT / numT << "\t\tavg     fiF:" << fiF / numF << endl;
			cout << "avg weightT:" << weightT / numT << "\t\tavg weightF:" << weigthF / numF << endl;
			if (1)
			{
				Mat_<uchar>tmpimg = images[augmented_images[maxImgIdx]].clone();
				for (int k = 0; k < current_shapes[maxImgIdx].rows; k++) {
					circle(tmpimg, Point(current_shapes[maxImgIdx](k, 0), current_shapes[maxImgIdx](k, 1)), 3, Scalar(255));
				}
				rectangle(tmpimg, Point((int)bounding_boxs[maxImgIdx].start_x, (int)bounding_boxs[maxImgIdx].start_y),
					Point((int)bounding_boxs[maxImgIdx].start_x + bounding_boxs[maxImgIdx].width,
					(int)bounding_boxs[maxImgIdx].start_y + bounding_boxs[maxImgIdx].height), Scalar(255));
				char tmpname[512];
				sprintf(tmpname, "%d_%d_%d_%d_%f.jpg", stages_, i, j, ground_truth_faces[maxImgIdx], current_weight[maxImgIdx]);
				imwrite(tmpname, tmpimg);
			}
			// sort fi with index
			vector<pair<float, int> > fiSort;
			fiSort.clear();
			for (int n = 0; n < current_fi.size(); ++n)
				fiSort.push_back(pair<float, int>(current_fi[n], n));
			sort(fiSort.begin(), fiSort.end(), my_cmp);
			// compute recall set threshold
			float max_recall = 0, min_error = 1;
			int idx_tmp = -1;
			vector<pair<float, float> > precise_recall;
			for (int n = 0; n < fiSort.size(); ++n)
			{
				int true_pos = 0; int false_neg = 0;
				int true_neg = 0; int false_pos = 0;
				for (int m = 0; m < fiSort.size(); ++m)
				{
					int isFace = ground_truth_faces[fiSort[m].second];
					// below the threshold as non-face
					if (m < n) {
						if (isFace == 1) {
							false_neg++;
						}
						else {
							true_neg++;
						}
					}
					else {
						if (isFace == 1) {
							true_pos++;
						}
						else {
							false_pos++;
						}
					}
				}
				if (true_pos / (true_pos + false_neg + FLT_MIN) >= max_recall) {
					max_recall = true_pos / (true_pos + false_neg + FLT_MIN);
					precise_recall.push_back(pair<float, float>(true_pos / (true_pos + false_neg + FLT_MIN), false_pos / (false_pos + true_neg + FLT_MIN)));
					rfs_[i][j].threshold = fiSort[n].first;
				}
				else
					break;
			}
			cout << "pre_recall in tree:" << precise_recall[precise_recall.size() - 1].first << "  " << precise_recall[precise_recall.size() - 1].second << endl;
			/////////////////////////////////////////////////////////////////////////////
			if (precise_recall[precise_recall.size() - 1].second < 0.5)
			{
				perfomtimes++;
				cout << "perform hard negative mining...." << perfomtimes << endl;
				int nn = posLenth*global_params.initial_num;
				int findtimes = 0;
				bool stop = false;
#pragma omp parallel
				{
					while (!stop)
					{
//						if (current_fi[nn] >= rfs_[i][j].threshold)
//						{
//#pragma omp critical		{
//							nn++;
//							continue;
//							}
//							
//						}
						findtimes++; 
						
						RNG random_generator(getTickCount());
						int idx = random_generator.uniform(posLenth, images.size() - 1);
						int idx2 = random_generator.uniform(0, posLenth - 1);
						BoundingBox new_box;
						getRandomBox(images[idx], bounding_boxs[idx2], new_box);
						Mat_<float> temp2 = ProjectShape(ground_truth_shapes[idx2], bounding_boxs[idx2]);
						Mat_<float> tmp_current_shapes = ReProjectShape(temp2, new_box);

						bool tmp_isface = true;
						float tmp_fi = 0;

						for (int s = 0; s <= stages; ++s) 
						{
							int iRange = RandomForest_[s].rfs_.size() - 1;
							if (s == stages) {
								iRange = i;
							}
							for (int r = 0; r <= iRange; ++r) {
								int jRange = RandomForest_[s].rfs_[r].size() - 1;
								if (r == i && s == stages) {
									jRange = j;
								}
								for (int t = 0; t <= jRange; ++t) {
									Mat_<float> rotation;
									float scale;
									SimilarityTransform(ProjectShape(tmp_current_shapes, new_box), mean_shape, rotation, scale);
									int bincode = 0;
									float score = 0;
									GetResultfromTree(RandomForest_[s].rfs_[r][t], images[idx], tmp_current_shapes, new_box, rotation, scale, &bincode, &score);
									tmp_fi += score;
									if (tmp_fi < RandomForest_[s].rfs_[r][t].threshold) {
										tmp_isface = false;
										break;
									}
								}
								if (!tmp_isface)break;
							}
							if (!tmp_isface)break;
							if ((s - 1) >= 0)
							{
								struct feature_node **binfeatures = new struct feature_node*[1];
								binfeatures[0] = new struct feature_node[RandomForest_[s - 1].max_numtrees_*RandomForest_[s - 1].num_landmark_ + 1];
								Mat_<float> rotation;
								float scale;
								SimilarityTransform(ProjectShape(tmp_current_shapes, new_box), mean_shape, rotation, scale);
								for (int j1 = 0; j1 < RandomForest_[s - 1].num_landmark_; j1++) {
									GetCodefromRandomForestOnlyOnce(binfeatures[0], j1*RandomForest_[s - 1].max_numtrees_,
										RandomForest_[s - 1].rfs_[j1], images[idx], tmp_current_shapes,
										new_box, rotation, scale);
								}
								binfeatures[0][RandomForest_[s - 1].num_landmark_ * RandomForest_[s - 1].max_numtrees_].index = -1;
								binfeatures[0][RandomForest_[s - 1].num_landmark_ * RandomForest_[s - 1].max_numtrees_].value = -1;
								GlobalRegressionOnlyOnce(binfeatures, tmp_current_shapes, new_box, mean_shape, Models_[s - 1]);
								delete[] binfeatures[0];
								delete[] binfeatures;
							}
						}
						if (tmp_isface) {
#pragma omp critical
							{
								if (nn < current_fi.size()) {
									augmented_images[nn] = idx;
									current_shapes[nn] = tmp_current_shapes;
									bounding_boxs[nn] = new_box;
									current_fi[nn] = tmp_fi;
									nn++;
								}
								else
									stop = true;
							}
						}
						/*if (findtimes > 1000000)
						{
							break;
						}*/
					}
				}
			}
			float time = float(clock() - ss_clk) / CLOCKS_PER_SEC;
			cout << "stage: " << stages << " landmark: " << i << " tree: " << j << " time:" << time << endl;
		}
		float time = float(clock() - tt_clk) / CLOCKS_PER_SEC;
		cout << "the train rf of " << i << "th landmark cost " << time << "s" << endl;
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

