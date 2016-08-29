#include "LBF.h"
#include "LBFRegressor.h"
using namespace std;
using namespace cv;

// parameters
Params global_params;
string modelPath = "model";
string dataPath = "regress";

int main( int argc, const char** argv ){
	
	InitializeGlobalParam();
	omp_set_num_threads(16);

	string listT="list1.txt";
	string listF="list2.txt";
	
	vector< Mat_<uchar> >		images_train;
	vector< Mat_<float> >		ground_truth_shapes_train;
	vector<int>					ground_truth_faces_train;
	vector<BoundingBox>			bounding_boxs_train;
	
	ifstream fin;locale::global(locale(""));fin.open(listT);locale::global(locale("C"));
	string line;
	int sample_freq=0;
	while(getline(fin,line)){
		sample_freq++;
		printf("loading train:%d\r",sample_freq);
		istringstream stream(line);
		string name;
		stream >> name;
		Mat_<uchar> image = imread(name,0);
		if (image.rows<MINHEIGHT||image.cols<MINHEIGHT||image.empty())
		{
			continue;
		}
		// Read ground truth shapes
		int shape28Index[28]={1,2,3,4,5,10,11,12,13,14,19,20,27,28,35,/*center 15th*/ 36 ,37,38,39,40,41,42,47,48,49,50,59,60};
		bool badshape = false;
        Mat_<float> ground_truth_shape(28,2);
		for (int i=0,j=0;i<95;++i){
			if ( (i+1) == shape28Index[j]){
				stream >> ground_truth_shape(j,0);
				stream >> ground_truth_shape(j,1);
				if (ground_truth_shape(j, 0)<0 || 
					ground_truth_shape(j, 1)<0 ||
					ground_truth_shape(j, 0)>=image.cols ||
					ground_truth_shape(j, 1)>=image.rows)
					badshape = true;
				++j;				
			}else{
				float tmp1,tmp2;
				stream >> tmp1;
				stream >> tmp2;
				/*if (tmp1 < 0 || tmp2 < 0 || tmp1 >= image.cols || tmp2 >= image.rows)
					badshape = true;*/
			}
		}
		if (badshape)
			continue;
		// Read Bounding box
		BoundingBox bbx = CalculateBoundingBox(image, ground_truth_shape);
		adjustImage(image, ground_truth_shape, bbx);

		images_train.push_back(image);
		ground_truth_shapes_train.push_back(ground_truth_shape);   
		ground_truth_faces_train.push_back(1);
		bounding_boxs_train.push_back(bbx);
	}
	fin.close();
	cout<<endl;
	int posLenth=images_train.size();
	////////////////////////////////////////////////////////////////////////////////
	ifstream fin2;locale::global(locale(""));fin2.open(listF);locale::global(locale("C"));
	sample_freq=0;
	while(getline(fin2,line)){
		istringstream stream(line);string name;stream >> name;
		FILE* imgFile;imgFile=fopen(name.c_str(),"rb");
		if (imgFile==NULL)
			cout<<"error in read data\n";
		else
		{
			while (1)
			{
				sample_freq++;
				printf("loading test:%d\r",sample_freq);
				int jWidth;
				int jHeight;
				fread(&jWidth,4,1,imgFile);
				fread(&jHeight,4,1,imgFile);
				uchar* jData = new uchar[jWidth*jHeight];
				fread(jData,1,jWidth*jHeight,imgFile);
				
				if (feof(imgFile)){
					delete[] jData;
					break;
				}
				if( jWidth<MINHEIGHT || jHeight<MINHEIGHT)
					continue;
				// Read Image
				Mat_<uchar> image(jHeight,jWidth);
				for (int k=0;k<jHeight;++k)
					for (int  l=0;l<jWidth;++l)
						image(k,l) = jData[k*jWidth+l];
				delete[] jData;

				if (image.rows > image.cols){
					if (image.rows > MAXHEIGHT_NEG){
						float scale = 1.0*MAXHEIGHT_NEG / image.rows;
						int new_width = image.cols*scale;
						int new_height = image.rows*scale;
						resize(image, image, Size(new_width, new_height));
					}
				}else
					if (image.cols > MAXHEIGHT_NEG){
						float scale = 1.0*MAXHEIGHT_NEG / image.cols;
						int new_width = image.cols*scale;
						int new_height = image.rows*scale;
						resize(image, image, Size(new_width, new_height));
					}
				BoundingBox new_box;
				RNG random_generator(getTickCount());
				int random_idx = random_generator.uniform(0, posLenth - 1);
				getRandomBox(image, bounding_boxs_train[random_idx], new_box);
				Mat_<float> temp = ProjectShape(ground_truth_shapes_train[random_idx], bounding_boxs_train[random_idx]);
				Mat_<float> ground_truth_shape = ReProjectShape(temp, new_box);
				images_train.push_back(image);
				ground_truth_shapes_train.push_back(ground_truth_shape);   
				ground_truth_faces_train.push_back(-1);
				bounding_boxs_train.push_back(new_box);
			}
		}
		fclose(imgFile);
	}
	fin2.close();
	cout<<"\ntrain num:"<<images_train.size()<<" ,pos is "<<posLenth<<endl;
	

	/*LBFRegressor regressor;
	regressor.Train(images_train, ground_truth_shapes_train, ground_truth_faces_train, bounding_boxs_train, posLenth);
	regressor.Save(modelPath);*/
	
	LBFRegressor regressor;
	regressor.Load(modelPath);

	Mat_<uchar> testImg=imread("img\\3.jpg",0);
	Mat_<uchar> forshow = testImg.clone();
	float scale=1.1;
	int minSize=40/scale;
	float shuffle=0.1;
	int ccc=0;
	int ttt=0;
	for (int i=0;i<15;++i)
	{
		minSize*=scale;
		for (int j=0;j<testImg.cols-minSize;j=j+minSize*shuffle)
		{
			for (int k=0;k<testImg.rows-minSize;k=k+minSize*shuffle)
			{
				ccc++;
				float left_x  = max(0,j);
				float top_y   = max(0,k);
				float right_x = min(testImg.cols-1,j+minSize);
				float bottom_y= min(testImg.rows-1,k+minSize);
				//cout<<left_x<<" "<<top_y<<" "<<right_x<<" "<<bottom_y<<endl;
				Mat_<uchar> tmp=testImg.rowRange((int)top_y,(int)bottom_y).colRange((int)left_x,(int)right_x).clone();
				
				BoundingBox bx;
				bx.start_x=0.3*minSize;
				bx.start_y=0.3*minSize;
				bx.width=(right_x-left_x)/1.6;
				bx.height=(bottom_y-top_y)/1.6;
				bx.centroid_x=(right_x-left_x)/2.0;
				bx.centroid_y=(bottom_y-top_y)/2.0;

				int count=0;
				bool result_face=true;
				Mat_<float>shapes = regressor.Predict(tmp,bx,1,result_face,count);
				if (result_face)
				{
					for (int m=0;m<shapes.rows;++m)
					{
						//circle(tmp,Point(shapes(m,0),shapes(m,1)),3,Scalar(255));
						circle(forshow,Point(left_x+shapes(m,0),top_y+shapes(m,1)),3,Scalar(255));
					}
					/*imshow("cur",tmp);
					waitKey(0);*/
				}
				else
					ttt+=count;
			}
		}
	}
	imshow("cur", forshow);waitKey(0);
	cout<<"ccc:"<<ccc<<endl;
	cout<<"ttt:"<<ttt<<endl;

	//
	//int right_num_pos=0,right_num_neg=0,pos_num=0,neg_num=0;
	//vector<Mat_<float>> current_shapes;
	//vector<Mat_<float>> real_shapes;
	//vector<int> real_idx;
	//int count=0;
	//for (int i=0;i<images_test.size();++i)
	//{
	//	bool result_face=true;
	//	int fcount=0;
	//	Mat_<float>shapes = regressor.Predict(images_test[i],bounding_boxs_test[i],1,result_face,fcount);
	//	if (!result_face)
	//	{
	//		count+=fcount;
	//	}
	//	
	//	if (1==ground_truth_faces_test[i] && result_face==true)
	//	{
	//		right_num_pos++;
	//	}
	//	if (-1==ground_truth_faces_test[i] && result_face==false)
	//	{
	//		right_num_neg++;
	//	}
	//	if (ground_truth_faces_test[i]==1 && result_face==true)
	//	{
	//		current_shapes.push_back(shapes);
	//		real_shapes.push_back(ground_truth_shapes_test[i]);
	//		real_idx.push_back(i);
	//	}
	//	if (ground_truth_faces_test[i]==1)
	//	{
	//		pos_num++;
	//	} 
	//	else
	//	{
	//		neg_num++;
	//	}
	//	/*if(ground_truth_faces_test[i]==1 && result_face==false)
	//	{
	//		Mat_<uchar> tmp=images_test[i].clone();
	//		for (int j=0;j<ground_truth_shapes_test[i].rows;++j)
	//		{
	//			circle(tmp,Point(ground_truth_shapes_test[i](j,0),ground_truth_shapes_test[i](j,1)),3,Scalar(255));
	//		}
	//		imshow("real",tmp);
	//		waitKey(0);

	//		Mat_<uchar> tmp2=images_test[i].clone();
	//		for (int j=0;j<shapes.rows;++j)
	//		{
	//			circle(tmp2,Point(shapes(j,0),shapes(j,1)),3,Scalar(255));
	//		}
	//		imshow("cur",tmp2);
	//		waitKey(0);
	//	}
	//	if(ground_truth_faces_test[i]==-1 && result_face==true)
	//	{
	//		Mat_<uchar> tmp=images_test[i].clone();
	//		for (int j=0;j<ground_truth_shapes_test[i].rows;++j)
	//		{
	//			circle(tmp,Point(ground_truth_shapes_test[i](j,0),ground_truth_shapes_test[i](j,1)),3,Scalar(255));
	//		}
	//		imshow("real",tmp);
	//		waitKey(0);

	//		Mat_<uchar> tmp2=images_test[i].clone();
	//		for (int j=0;j<shapes.rows;++j)
	//		{
	//			circle(tmp2,Point(shapes(j,0),shapes(j,1)),3,Scalar(255));
	//		}
	//		imshow("cur",tmp2);
	//		waitKey(0);
	//	}*/
	//	
	//}
	//cout<<"average compare in false samples:"<<count*1.0/neg_num<<endl;
	//float classification=(right_num_neg+right_num_pos)*1.0/images_test.size();
	////cout<<"total right rate:"<<classification<<endl;
	//cout<<"pos right rate:"<<right_num_pos*1.0/pos_num<<endl;
	//cout<<"neg error rate:"<<1-right_num_neg*1.0/neg_num<<endl;
	//float MRSE_sum = 0;
	//for (int i =0; i<real_idx.size();i++){
	//	MRSE_sum += CalculateError(real_shapes[i], current_shapes[i]);
	//	
	//	Mat_<uchar> tmp=images_test[real_idx[i]].clone();
	//	for (int j=0;j<real_shapes[i].rows;++j)
	//	{
	//		circle(tmp,Point(real_shapes[i](j,0),real_shapes[i](j,1)),3,Scalar(255));
	//	}
	//	imshow("real",tmp);
	//	waitKey(0);

	//	Mat_<uchar> tmp2=images_test[real_idx[i]].clone();
	//	for (int j=0;j<current_shapes[i].rows;++j)
	//	{
	//		circle(tmp2,Point(current_shapes[i](j,0),current_shapes[i](j,1)),3,Scalar(255));
	//	}
	//	imshow("cur",tmp2);
	//	waitKey(0);
	//}
	//cout << "Mean Root Square Error is "<< MRSE_sum/current_shapes.size()*100 <<"%"<<endl;

	system("pause");
	return 0;
}

void InitializeGlobalParam(){
    global_params.bagging_overlap = 0.4;
    global_params.max_numtrees = 10;
    global_params.max_depth = 4;
    global_params.landmark_num = 28;
    global_params.initial_num = 1;
    global_params.max_numstage = 5;

    float m_max_radio_radius[10] = {1.0, 0.8, 0.4, 0.2, 0.1, 0.10, 0.08, 0.06, 0.06,0.05};
    float m_max_numfeats[10] = {2000, 2000, 2000, 2000, 2000, 300, 200,200,100,100};
	float m_max_probility[10] = {0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0};

    for (int i=0;i<10;i++){
        global_params.max_radio_radius[i] = m_max_radio_radius[i];
    }
    for (int i=0;i<10;i++){
        global_params.max_numfeats[i] = m_max_numfeats[i];
    }
	for (int i=0;i<10;i++){
		global_params.max_probility[i] = m_max_probility[i];
	}
    global_params.max_numthreshs = 24;
}

void ReadGlobalParamFromFile(string path){
    cout << "Loading GlobalParam..." << endl;
    ifstream fin;
	locale::global(locale(""));
    fin.open(path);
	locale::global(locale("C"));
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
    cout << "Loading GlobalParam end"<<endl;
    fin.close();
}

