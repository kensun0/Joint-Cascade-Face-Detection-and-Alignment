//
//  main.cpp
//  myopencv
//
//  Created by lequan on 1/24/15.
//  Copyright (c) 2015 lequan. All rights reserved.
//

#include "LBF.h"
#include "LBFRegressor.h"
//#include "implement.h"
using namespace std;
using namespace cv;

// parameters
Params global_params;


string modelPath ="F:\\";
string dataPath = "F:\\";
string cascadeName = "haarcascade_frontalface_alt.xml";   // no used

void InitializeGlobalParam();
void PrintHelp();

int main( int argc, const char** argv ){
	InitializeGlobalParam();
	
	string listT="F:\\listT.txt";       //address of postive samples
	string listF="F:\\listF.txt";       //address of negative samples
	
	vector<Mat_<uchar> > images_train;
	vector<Mat_<float> > ground_truth_shapes_train;
	vector<int> ground_truth_faces_train;
	vector<BoundingBox> bounding_boxs_train;
	
	vector<Mat_<uchar> > images_test;
	vector<Mat_<float> > ground_truth_shapes_test;
	vector<int> ground_truth_faces_test;
	vector<BoundingBox> bounding_boxs_test;

	ifstream fin;
	locale::global(locale(""));    //only for Chinese 
	fin.open(listT);
	locale::global(locale("C"));
	string line;
	int sample_freq=0;
	while(getline(fin,line)){
		
		sample_freq++;
		printf("loading train:%d\r",sample_freq);
		/*if (sample_freq%4!=0)
		{
			continue;
		}
		if (sample_freq>=200)
		{
			break;
		}*/
		istringstream stream(line);
		string name;
		stream >> name;
		// Read Image
		Mat_<uchar> image = imread(name,0);
		if (image.rows<10||image.cols<10)
		{
			continue;
		}
		// Read ground truth shapes, you should rewrite it according to your samples
		int shape28Index[28]={1,2,3,4,5,
							10,11,12,13,14,
							19,20,
							27,28,
							35,36,
							37,38,
							39,40,
							41,42,
							47,48,
							49,50,
							59,60};
        Mat_<float> ground_truth_shape(28,2);
    // we use 28 keypoints of 95 keypoints 
		for (int i=0,j=0;i<95;++i)
		{
			if ( (i+1) == shape28Index[j])
			{
				stream >> ground_truth_shape(j,0);
				stream >> ground_truth_shape(j,1);
				//cout<<ground_truth_shape<<endl;
				++j;				
			}
			else
			{
				float tmp;
				stream >> tmp;
				stream >> tmp;
			}
		}
		// Read Bounding box
		BoundingBox bbx = CalculateBoundingBox2(ground_truth_shape);
		adjustImage2(image,ground_truth_shape,bbx);
		RNG random_generator(getTickCount());
		if (random_generator.uniform(0.0,1.0)<(0.9))
		{
			images_train.push_back(image);
			ground_truth_shapes_train.push_back(ground_truth_shape);   
			ground_truth_faces_train.push_back(1);
			bounding_boxs_train.push_back(bbx);
			/*for(int i=0;i<ground_truth_shape.rows;i++){
				circle(image,Point(ground_truth_shape(i,0),ground_truth_shape(i,1)),3,Scalar(255));
			}
			imshow("test2",image);
			waitKey(0);*/
		} 
		else
		{
			images_test.push_back(image);
			ground_truth_shapes_test.push_back(ground_truth_shape);   
			ground_truth_faces_test.push_back(1);
			bounding_boxs_test.push_back(bbx);
			//rectangle(image,Point(new_box.start_x,new_box.start_y),Point(new_box.start_x+new_box.width,new_box.start_y+new_box.height),Scalar(255));
			
		}


	}
	fin.close();
	cout<<endl;
	int posLenth=images_train.size();
	////////////////////////////////////////////////////////////////////////////////
	ifstream fin2;
	locale::global(locale(""));
	fin2.open(listF);
	locale::global(locale("C"));
	line;
	sample_freq=0;
	while(getline(fin2,line)){

		sample_freq++;
		printf("loading test:%d\r",sample_freq);
		/*if (sample_freq%100!=0)
		{
			continue;
		}*/
		/*if (sample_freq>=500)
		{
			break;
		}*/

		istringstream stream(line);
		string name;
		stream >> name;
		// Read Image
		Mat_<uchar> image = imread(name,0);
		
		// due to memory size, we resize big img to small
		if (image.rows>image.cols)
		{
			if (image.rows>MAXHEIGHT2)
			{
				float scale=1.0*MAXHEIGHT2/image.rows;
				int new_width=image.cols*scale;
				int new_height=image.rows*scale;
				resize(image,image,Size(new_width,new_height));
			} 
		} 
		else
		{
			if (image.cols>MAXHEIGHT2)
			{
				float scale=1.0*MAXHEIGHT2/image.cols;
				int new_width=image.cols*scale;
				int new_height=image.rows*scale;
				resize(image,image,Size(new_width,new_height));
			}
		}
		
		// Read ground truth shapes
		RNG random_generator(getTickCount());
		int random_idx = random_generator.uniform(0,posLenth-1);
		// random box
		BoundingBox new_box;
		int img_height=image.rows;
		int img_width=image.cols;

		int box_height=img_height;
		int box_width=img_width;
		do 
		{
			if (img_height>MINHEIGHT)
			{
				box_height= random_generator.uniform(MINHEIGHT*1.0,(double)img_height*0.8);
			}
			box_width = box_height*bounding_boxs_train[random_idx].width/bounding_boxs_train[random_idx].height;
		} while (box_width > img_width*0.8);
		
		int box_x = random_generator.uniform(0.0+0.2*img_width,(double)img_width*0.8-box_width);
		int box_y = random_generator.uniform(0.0+0.2*img_height,(double)img_height*0.8-box_height);
		new_box.start_x=box_x;
		new_box.start_y=box_y;
		new_box.width=box_width;
		new_box.height=box_height;
		new_box.centroid_x=box_x+box_width/2.0;
		new_box.centroid_y=box_y+box_height/2.0;


		Mat_<float> temp = ProjectShape(ground_truth_shapes_train[random_idx], bounding_boxs_train[random_idx]);
		Mat_<float> ground_truth_shape = ReProjectShape(temp, new_box);

		/*rectangle(image,Point(new_box.start_x,new_box.start_y),Point(new_box.start_x+new_box.width,new_box.start_y+new_box.height),Scalar(255));
		for(int i=0;i<ground_truth_shape.rows;i++){
			circle(image,Point(ground_truth_shape(i,0),ground_truth_shape(i,1)),3,Scalar(255));
		}
		imshow("test2",image);
		waitKey(0);*/

		if (random_generator.uniform(0.0,1.0)<(0.9))
		{
			images_train.push_back(image);
			ground_truth_shapes_train.push_back(ground_truth_shape);   
			ground_truth_faces_train.push_back(-1);
			bounding_boxs_train.push_back(new_box);
		} 
		else
		{
			images_test.push_back(image);
			ground_truth_shapes_test.push_back(ground_truth_shape);   
			ground_truth_faces_test.push_back(-1);
			bounding_boxs_test.push_back(new_box);
		}
	}
	fin2.close();

	cout<<endl;

	cout<<"train num:"<<images_train.size()<<" ,pos is "<<posLenth<<endl;
	cout<<"test num:"<<images_test.size()<<endl;

	/*LBFRegressor regressor;
	regressor.Train(images_train,ground_truth_shapes_train,ground_truth_faces_train,bounding_boxs_train);
	regressor.Save(modelPath+"LBF6.model");*/
	
	LBFRegressor regressor;
	regressor.Load(modelPath+"LBF6.model");

  // have not finished the test operation
	Mat_<uchar> testImg=imread("F:\\test.jpg",0);
	float scale=1.1;
	int minSize=50/scale;
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
					/*for (int m=0;m<shapes.rows;++m)
					{
						circle(tmp,Point(shapes(m,0),shapes(m,1)),3,Scalar(255));
					}
					imshow("cur",tmp);
					waitKey(0);*/
				}
				else
					ttt+=count;
			}
		}
	}
	cout<<"ccc:"<<ccc<<endl;
	cout<<"ttt:"<<ttt<<endl;

	
	int right_num_pos=0,right_num_neg=0,pos_num=0,neg_num=0;
	vector<Mat_<float>> current_shapes;
	vector<Mat_<float>> real_shapes;
	vector<int> real_idx;
	int count=0;
	for (int i=0;i<images_test.size();++i)
	{
		bool result_face=true;
		int fcount=0;
		Mat_<float>shapes = regressor.Predict(images_test[i],bounding_boxs_test[i],1,result_face,fcount);
		if (!result_face)
		{
			count+=fcount;
		}
		
		if (1==ground_truth_faces_test[i] && result_face==true)
		{
			right_num_pos++;
		}
		if (-1==ground_truth_faces_test[i] && result_face==false)
		{
			right_num_neg++;
		}
		if (ground_truth_faces_test[i]==1 && result_face==true)
		{
			current_shapes.push_back(shapes);
			real_shapes.push_back(ground_truth_shapes_test[i]);
			real_idx.push_back(i);
		}
		if (ground_truth_faces_test[i]==1)
		{
			pos_num++;
		} 
		else
		{
			neg_num++;
		}
		/*if(ground_truth_faces_test[i]==1 && result_face==false)
		{
			Mat_<uchar> tmp=images_test[i].clone();
			for (int j=0;j<ground_truth_shapes_test[i].rows;++j)
			{
				circle(tmp,Point(ground_truth_shapes_test[i](j,0),ground_truth_shapes_test[i](j,1)),3,Scalar(255));
			}
			imshow("real",tmp);
			waitKey(0);

			Mat_<uchar> tmp2=images_test[i].clone();
			for (int j=0;j<shapes.rows;++j)
			{
				circle(tmp2,Point(shapes(j,0),shapes(j,1)),3,Scalar(255));
			}
			imshow("cur",tmp2);
			waitKey(0);
		}
		if(ground_truth_faces_test[i]==-1 && result_face==true)
		{
			Mat_<uchar> tmp=images_test[i].clone();
			for (int j=0;j<ground_truth_shapes_test[i].rows;++j)
			{
				circle(tmp,Point(ground_truth_shapes_test[i](j,0),ground_truth_shapes_test[i](j,1)),3,Scalar(255));
			}
			imshow("real",tmp);
			waitKey(0);

			Mat_<uchar> tmp2=images_test[i].clone();
			for (int j=0;j<shapes.rows;++j)
			{
				circle(tmp2,Point(shapes(j,0),shapes(j,1)),3,Scalar(255));
			}
			imshow("cur",tmp2);
			waitKey(0);
		}*/
		
	}
	cout<<"average compare in false samples:"<<count*1.0/neg_num<<endl;
	float classification=(right_num_neg+right_num_pos)*1.0/images_test.size();
	//cout<<"total right rate:"<<classification<<endl;
	cout<<"pos right rate:"<<right_num_pos*1.0/pos_num<<endl;
	cout<<"neg error rate:"<<1-right_num_neg*1.0/neg_num<<endl;
	float MRSE_sum = 0;
	for (int i =0; i<real_idx.size();i++){
		MRSE_sum += CalculateError(real_shapes[i], current_shapes[i]);
		
		Mat_<uchar> tmp=images_test[real_idx[i]].clone();
		for (int j=0;j<real_shapes[i].rows;++j)
		{
			circle(tmp,Point(real_shapes[i](j,0),real_shapes[i](j,1)),3,Scalar(255));
		}
		imshow("real",tmp);
		waitKey(0);

		Mat_<uchar> tmp2=images_test[real_idx[i]].clone();
		for (int j=0;j<current_shapes[i].rows;++j)
		{
			circle(tmp2,Point(current_shapes[i](j,0),current_shapes[i](j,1)),3,Scalar(255));
		}
		imshow("cur",tmp2);
		waitKey(0);
	}
	cout << "Mean Root Square Error is "<< MRSE_sum/current_shapes.size()*100 <<"%"<<endl;

	system("pause");
	return 0;
}

void InitializeGlobalParam(){
    global_params.bagging_overlap = 0.4;
    global_params.max_numtrees = 10;
    global_params.max_depth = 6;
    global_params.landmark_num = 28;
    global_params.initial_num = 1;
    
    global_params.max_numstage = 5;
    float m_max_radio_radius[10] = {0.4+0.,0.3+0.,0.2+0.,0.15+0.0, 0.12+0.0, 0.10+0.0, 0.08+0.0, 0.06+0.0, 0.06+0.0,0.05+0.0};
    float m_max_numfeats[10] = {1000+00, 1000+00, 1000+00, 1000+00, 1000+00, 300, 200,200,100,100};
	float m_max_probility[10] = {0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0};
	//float m_max_probility[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0};

    for (int i=0;i<10;i++){
        global_params.max_radio_radius[i] = m_max_radio_radius[i];
    }
    for (int i=0;i<10;i++){
        global_params.max_numfeats[i] = m_max_numfeats[i];
    }
	for (int i=0;i<10;i++){
		global_params.max_probility[i] = m_max_probility[i];
	}
    global_params.max_numthreshs = 2000;
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
void PrintHelp(){
    cout << "Useage:"<<endl;
    cout << "1. train your own model:    LBF.out  TrainModel "<<endl;
    cout << "2. test model on dataset:   LBF.out  TestModel"<<endl;
    cout << "3. test model via a camera: LBF.out  Demo "<<endl;
    cout << "4. test model on a pic:     LBF.out  Demo xx.jpg"<<endl;
    cout << "5. test model on pic set:   LBF.out  Demo Img_Path.txt"<<endl;
    cout << endl;
}
