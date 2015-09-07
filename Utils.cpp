#include "LBF.h"
#include "LBFRegressor.h"
using namespace std;
using namespace cv;

Mat_<float> GetMeanShape(const vector<Mat_<float> >& shapes,
                          const vector<BoundingBox>& bounding_box){
    Mat_<float> result = Mat::zeros(shapes[0].rows,2,CV_32FC1);
    for(int i = 0;i < shapes.size();i++){
       
        result = result + ProjectShape(shapes[i],bounding_box[i]);
    }
    result = 1.0 / shapes.size() * result;
    return result;
}

Mat_<float> GetMeanShape2(const vector<Mat_<float> >& shapes,
	const vector<BoundingBox>& bounding_box, const vector<int>& ground_truth_faces){
		Mat_<float> result = Mat::zeros(shapes[0].rows,2,CV_32FC1);
		int tmp=0;
		for(int i = 0;i < shapes.size();i++){
			if (ground_truth_faces[i]==1)
			{
				result = result + ProjectShape(shapes[i],bounding_box[i]);
				tmp++;
			}
		}
		result = 1.0 / tmp * result;
		return result;
}

void GetShapeResidual(const vector<Mat_<float> >& ground_truth_shapes,
                      const vector<Mat_<float> >& current_shapes,
                      const vector<BoundingBox>& bounding_boxs,
                      const Mat_<float>& mean_shape,
                      vector<Mat_<float> >& shape_residuals){
    
    Mat_<float> rotation;
    float scale;
    shape_residuals.resize(bounding_boxs.size());
    for (int i = 0;i < bounding_boxs.size(); i++){
        shape_residuals[i] = ProjectShape(ground_truth_shapes[i], bounding_boxs[i])
        - ProjectShape(current_shapes[i], bounding_boxs[i]);
        SimilarityTransform(mean_shape, ProjectShape(current_shapes[i],bounding_boxs[i]),rotation,scale);
        transpose(rotation,rotation);
        shape_residuals[i] = scale * shape_residuals[i] * rotation;
    }
}


Mat_<float> ProjectShape(const Mat_<float>& shape, const BoundingBox& bounding_box){
    Mat_<float> temp(shape.rows,2);
    for(int j = 0;j < shape.rows;j++){
        temp(j,0) = (shape(j,0)-bounding_box.centroid_x) / (bounding_box.width / 2.0);
        temp(j,1) = (shape(j,1)-bounding_box.centroid_y) / (bounding_box.height / 2.0);  
    } 
    return temp;  
}

Mat_<float> ReProjectShape(const Mat_<float>& shape, const BoundingBox& bounding_box){
    Mat_<float> temp(shape.rows,2);
    for(int j = 0;j < shape.rows;j++){
        temp(j,0) = (shape(j,0) * bounding_box.width / 2.0 + bounding_box.centroid_x);
        temp(j,1) = (shape(j,1) * bounding_box.height / 2.0 + bounding_box.centroid_y);
    } 
    return temp; 
}


void SimilarityTransform(const Mat_<float>& shape1, const Mat_<float>& shape2, 
                         Mat_<float>& rotation,float& scale){
    rotation = Mat::zeros(2,2,CV_32FC1);
    scale = 0;
    
    // center the data
    float center_x_1 = 0;
    float center_y_1 = 0;
    float center_x_2 = 0;
    float center_y_2 = 0;
    for(int i = 0;i < shape1.rows;i++){
        center_x_1 += shape1(i,0);
        center_y_1 += shape1(i,1);
        center_x_2 += shape2(i,0);
        center_y_2 += shape2(i,1); 
    }
    center_x_1 /= shape1.rows;
    center_y_1 /= shape1.rows;
    center_x_2 /= shape2.rows;
    center_y_2 /= shape2.rows;
    
    Mat_<float> temp1 = shape1.clone();
    Mat_<float> temp2 = shape2.clone();
    for(int i = 0;i < shape1.rows;i++){
        temp1(i,0) -= center_x_1;
        temp1(i,1) -= center_y_1;
        temp2(i,0) -= center_x_2;
        temp2(i,1) -= center_y_2;
    }

     
    Mat_<float> covariance1, covariance2;
    Mat_<float> mean1,mean2;
    // calculate covariance matrix
    calcCovarMatrix(temp1,covariance1,mean1,CV_COVAR_SCALE|CV_COVAR_ROWS|CV_COVAR_NORMAL,CV_32F);
	calcCovarMatrix(temp2,covariance2,mean2,CV_COVAR_SCALE|CV_COVAR_ROWS|CV_COVAR_NORMAL,CV_32F);
	//cout<<covariance1<<endl;
	//cout<<covariance2<<endl;
    float s1 = sqrt(norm(covariance1));
    float s2 = sqrt(norm(covariance2));
    scale = s1 / s2; 
    temp1 = 1.0 / s1 * temp1;
    temp2 = 1.0 / s2 * temp2;

    float num = 0;
    float den = 0;
    for(int i = 0;i < shape1.rows;i++){
        num = num + temp1(i,1) * temp2(i,0) - temp1(i,0) * temp2(i,1);
        den = den + temp1(i,0) * temp2(i,0) + temp1(i,1) * temp2(i,1);      
    }
    
    float norm = sqrt(num*num + den*den);    
    float sin_theta = num / norm;
    float cos_theta = den / norm;
    rotation(0,0) = cos_theta;
    rotation(0,1) = -sin_theta;
    rotation(1,0) = sin_theta;
    rotation(1,1) = cos_theta;
}

float calculate_covariance(const vector<float>& v_1, 
                            const vector<float>& v_2){
    Mat_<float> v1(v_1);
    Mat_<float> v2(v_2);
    float mean_1 = mean(v1)[0];
    float mean_2 = mean(v2)[0];
    v1 = v1 - mean_1;
    v2 = v2 - mean_2;
    return mean(v1.mul(v2))[0]; 
}
Mat_<float> LoadGroundTruthShape(string& filename){
    Mat_<float> shape(global_params.landmark_num,2);
    ifstream fin;
    string temp;
    locale::global(locale(""));
    fin.open(filename);
	locale::global(locale("C"));
    getline(fin, temp);
    getline(fin, temp);
    getline(fin, temp);
    //for (int i=0;i<global_params.landmark_num;i++){
	int k=0;
	for (int i=0;i<68;i++){
		float tmp;
		int ppp[29]={9,18,20,22,23,25,27,31,32,34,36,37,38,39,40,41,42,43,44,45,46,47,48,49,52,55,58,63,67};
		bool ttt=false;
		for (int j=0;j<29;++j)
		{
			if(ppp[j]==(i+1)){
				ttt=true;
			}
		}
		if (ttt)
		{fin >> shape(k,0) >> shape(k,1);k++;
		} 
		else
		{fin >> tmp >>tmp;
		}
		//fin >> shape(i,0) >> shape(i,1);
    }
    fin.close();
    return shape;
    
}
BoundingBox CalculateBoundingBox(Mat_<float>& shape){
    BoundingBox bbx;
    float left_x = 10000;
    float right_x = 0;
    float top_y = 10000;
    float bottom_y = 0;
    for (int i=0; i < shape.rows;i++){
        if (shape(i,0) < left_x)
            left_x = shape(i,0);
        if (shape(i,0) > right_x)
            right_x = shape(i,0);
        if (shape(i,1) < top_y)
            top_y = shape(i,1);
        if (shape(i,1) > bottom_y)
            bottom_y = shape(i,1);
    }
    bbx.start_x = left_x;
    bbx.start_y = top_y;
    bbx.height  = bottom_y - top_y;
    bbx.width   = right_x - left_x;
    bbx.centroid_x = bbx.start_x + bbx.width/2.0;
    bbx.centroid_y = bbx.start_y + bbx.height/2.0;
    return bbx;
}

void adjustImage(Mat_<uchar>& img,
                 Mat_<float>& ground_truth_shape,
                 BoundingBox& bounding_box){
    float left_x  = max(1.0, (double)bounding_box.centroid_x - bounding_box.width*2/3);
    float top_y   = max(1.0, (double)bounding_box.centroid_y - bounding_box.height*2/3);
    float right_x = min(img.cols-1.0,(double)bounding_box.centroid_x+bounding_box.width);
    float bottom_y= min(img.rows-1.0,(double)bounding_box.centroid_y+bounding_box.height);
    img = img.rowRange((int)top_y,(int)bottom_y).colRange((int)left_x,(int)right_x).clone();
	/*imshow("test",img);
	waitKey(0);*/
    bounding_box.start_x = bounding_box.start_x-left_x;
    bounding_box.start_y = bounding_box.start_y-top_y;
    bounding_box.centroid_x = bounding_box.start_x + bounding_box.width/2.0;
    bounding_box.centroid_y = bounding_box.start_y + bounding_box.height/2.0;
    
    for(int i=0;i<ground_truth_shape.rows;i++){
        ground_truth_shape(i,0) = ground_truth_shape(i,0)-left_x;
        ground_truth_shape(i,1) = ground_truth_shape(i,1)-top_y;
    }
}

BoundingBox CalculateBoundingBox2(Mat_<float>& shape){
    BoundingBox bbx;
    float left_x = 10000;
    float right_x = 0;
    float top_y = 10000;
    float bottom_y = 0;
    for (int i=0; i < shape.rows;i++){
        if (shape(i,0) < left_x)
            left_x = shape(i,0);
        if (shape(i,0) > right_x)
            right_x = shape(i,0);
        if (shape(i,1) < top_y)
            top_y = shape(i,1);
        if (shape(i,1) > bottom_y)
            bottom_y = shape(i,1);
    }
    bbx.start_x = left_x;
    bbx.start_y = top_y;
    bbx.height  = bottom_y - top_y;
    bbx.width   = right_x - left_x;
    bbx.centroid_x = bbx.start_x + bbx.width/2.0;
    bbx.centroid_y = bbx.start_y + bbx.height/2.0;
    return bbx;
}

void adjustImage2(Mat_<uchar>& img,
                 Mat_<float>& ground_truth_shape,
                 BoundingBox& bounding_box){
	/*imshow("test",img);
	waitKey(0);*/
    float left_x  = max(1.0, (double)bounding_box.centroid_x - bounding_box.width*0.8);
    float top_y   = max(1.0, (double)bounding_box.centroid_y - bounding_box.height*0.8);
    float right_x = min(img.cols-1.0,(double)bounding_box.centroid_x+bounding_box.width*0.8);
    float bottom_y= min(img.rows-1.0,(double)bounding_box.centroid_y+bounding_box.height*0.8);
    img = img.rowRange((int)top_y,(int)bottom_y).colRange((int)left_x,(int)right_x).clone();
	
    bounding_box.start_x = bounding_box.start_x-left_x;
    bounding_box.start_y = bounding_box.start_y-top_y;
    bounding_box.centroid_x = bounding_box.start_x + bounding_box.width/2.0;
    bounding_box.centroid_y = bounding_box.start_y + bounding_box.height/2.0;
    
    for(int i=0;i<ground_truth_shape.rows;i++){
        ground_truth_shape(i,0) = ground_truth_shape(i,0)-left_x;
        ground_truth_shape(i,1) = ground_truth_shape(i,1)-top_y;
    }
//imshow("test1",img);
//		waitKey(0);
	float ori_height=img.rows;
	float ori_weight=img.cols;

	if (ori_height>MAXHEIGHT)
	{
		float scale=MAXHEIGHT/ori_height;
		resize(img,img,Size(ori_weight*scale,ori_height*scale));
		bounding_box.start_x*=scale;
		bounding_box.start_y*=scale;
		bounding_box.centroid_x*=scale;
		bounding_box.centroid_y*=scale;
		bounding_box.width*=scale;
		bounding_box.height*=scale;
		for(int i=0;i<ground_truth_shape.rows;i++){
			ground_truth_shape(i,0) *= scale;
			ground_truth_shape(i,1) *= scale;
		}
	} 
	/*rectangle(img,Point(bounding_box.start_x,bounding_box.start_y),Point(bounding_box.start_x+bounding_box.width,bounding_box.start_y+bounding_box.height),Scalar(255));
	for(int i=0;i<ground_truth_shape.rows;i++){
		circle(img,Point(ground_truth_shape(i,0),ground_truth_shape(i,1)),3,Scalar(255));
	}
	imshow("test2",img);
	waitKey(0);*/
}

void LoadData(string filepath,
              vector<Mat_<uchar> >& images,
              vector<Mat_<float> >& ground_truth_shapes,
              vector<BoundingBox> & bounding_boxs
              ){
    ifstream fin;
	locale::global(locale(""));
    fin.open(filepath);
	locale::global(locale("C"));
    string name;
    while(getline(fin,name)){
        name.erase(0, name.find_first_not_of(" \t"));
        name.erase(name.find_last_not_of(" \t") + 1);
        cout << "file:" << name <<endl;
        
        // Read Image
        Mat_<uchar> image = imread(name,0);
        images.push_back(image);
        
        // Read ground truth shapes
        name.replace(name.find_last_of("."), 4,".pts");
        Mat_<float> ground_truth_shape = LoadGroundTruthShape(name);
        ground_truth_shapes.push_back(ground_truth_shape);
            
        // Read Bounding box
        BoundingBox bbx = CalculateBoundingBox(ground_truth_shape);
        bounding_boxs.push_back(bbx);
    }
    fin.close();
}

bool IsShapeInRect(Mat_<float>& shape, Rect& rect,float scale){
    float sum1 = 0;
    float sum2 = 0;
    float max_x=0,min_x=10000,max_y=0,min_y=10000;
    for (int i= 0;i < shape.rows;i++){
        if (shape(i,0)>max_x) max_x = shape(i,0);
        if (shape(i,0)<min_x) min_x = shape(i,0);
        if (shape(i,1)>max_y) max_y = shape(i,1);
        if (shape(i,1)<min_y) min_y = shape(i,1);
        
        sum1 += shape(i,0);
        sum2 += shape(i,1);
    }
    if ((max_x-min_x)>rect.width*1.5){
        return false;
    }
    if ((max_y-min_y)>rect.height*1.5){
        return false;
    }
    if (abs(sum1/shape.rows - (rect.x+rect.width/2.0)*scale) > rect.width*scale/2.0){
        return false;
    }
    if (abs(sum2/shape.rows - (rect.y+rect.height/2.0)*scale) > rect.height*scale/2.0){
        return false;
    }
    return true;
}

void LoadOpencvBbxData(string filepath,
                       vector<Mat_<uchar> >& images,
                       vector<Mat_<float> >& ground_truth_shapes,
                       vector<BoundingBox> & bounding_boxs
              ){
    ifstream fin;
	locale::global(locale(""));
    fin.open(filepath);
	locale::global(locale("C"));
    CascadeClassifier cascade;
    float scale = 1.3;
    extern string cascadeName;
    vector<Rect> faces;
    Mat gray;
    
    // --Detection
    cascade.load(cascadeName);
    string name;
    while(getline(fin,name)){
        name.erase(0, name.find_first_not_of(" \t"));
        name.erase(name.find_last_not_of(" \t") + 1);
        cout << "file:" << name <<endl;
        // Read Image
        Mat_<uchar> image = imread(name,0);
        
        
        // Read ground truth shapes
        name.replace(name.find_last_of("."), 4,".pts");
        Mat_<float> ground_truth_shape = LoadGroundTruthShape(name);
        
        // Read OPencv Detection Bbx
        Mat smallImg( cvRound (image.rows/scale), cvRound(image.cols/scale), CV_8UC1 );
        resize( image, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
        equalizeHist( smallImg, smallImg );
        
        // --Detection
        cascade.detectMultiScale( smallImg, faces,
                                 1.1, 2, 0
                                 //|CV_HAAR_FIND_BIGGEST_OBJECT
                                 //|CV_HAAR_DO_ROUGH_SEARCH
                                 |CV_HAAR_SCALE_IMAGE
                                 ,
                                 Size(30, 30) );
        for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++){
            Rect rect = *r;
            if (IsShapeInRect(ground_truth_shape,rect,scale)){
                Point center;
                BoundingBox boundingbox;
                
                boundingbox.start_x = r->x*scale;
                boundingbox.start_y = r->y*scale;
                boundingbox.width   = (r->width-1)*scale;
                boundingbox.height  = (r->height-1)*scale;
                boundingbox.centroid_x = boundingbox.start_x + boundingbox.width/2.0;
                boundingbox.centroid_y = boundingbox.start_y + boundingbox.height/2.0;
                
                
                adjustImage(image,ground_truth_shape,boundingbox);
                images.push_back(image);
                ground_truth_shapes.push_back(ground_truth_shape);
                bounding_boxs.push_back(boundingbox);
//                // add train data
//                bounding_boxs.push_back(boundingbox);
//                images.push_back(image);
//                ground_truth_shapes.push_back(ground_truth_shape);
                
//                rectangle(image, cvPoint(boundingbox.start_x,boundingbox.start_y),
//                          cvPoint(boundingbox.start_x+boundingbox.width,boundingbox.start_y+boundingbox.height),Scalar(0,255,0), 1, 8, 0);
//                for (int i = 0;i<ground_truth_shape.rows;i++){
//                    circle(image,Point2d(ground_truth_shape(i,0),ground_truth_shape(i,1)),1,Scalar(255,0,0),-1,8,0);
//
//                }
//                imshow("BBX",image);
//                cvWaitKey(0);
                break;
            }
        }
    }
    fin.close();
}
float CalculateError(const Mat_<float>& ground_truth_shape, const Mat_<float>& predicted_shape){
    Mat_<float> temp;
    //temp = ground_truth_shape.rowRange(36, 37)-ground_truth_shape.rowRange(45, 46);
	temp = ground_truth_shape.rowRange(1, 2)-ground_truth_shape.rowRange(6, 7);
    float x =mean(temp.col(0))[0];
    float y = mean(temp.col(1))[1];
    float interocular_distance = sqrt(x*x+y*y);
    float sum = 0;
    for (int i=0;i<ground_truth_shape.rows;i++){
        sum += norm(ground_truth_shape.row(i)-predicted_shape.row(i));
    }
    return sum/(ground_truth_shape.rows*interocular_distance);
}

void LoadCofwTrainData(vector<Mat_<uchar> >& images,
                       vector<Mat_<float> >& ground_truth_shapes,
                       vector<BoundingBox>& bounding_boxs){
    int img_num = 1345;
    cout<<"Read images..."<<endl;
    for(int i = 0;i < img_num;i++){
        string image_name = "/Users/lequan/workspace/xcode/myopencv/COFW_Dataset/trainingImages/";
        image_name = image_name + to_string((_Longlong)i+1) + ".jpg";
        Mat_<uchar> temp = imread(image_name,0);
        images.push_back(temp);
    }
    
    ifstream fin;
	locale::global(locale(""));
    fin.open("/Users/lequan/workspace/xcode/myopencv/COFW_Dataset/boundingbox.txt");
	locale::global(locale("C"));
    for(int i = 0;i < img_num;i++){
        BoundingBox temp;
        fin>>temp.start_x>>temp.start_y>>temp.width>>temp.height;
        temp.centroid_x = temp.start_x + temp.width/2.0;
        temp.centroid_y = temp.start_y + temp.height/2.0; 
        bounding_boxs.push_back(temp);
    }
    fin.close(); 
	locale::global(locale(""));
    fin.open("/Users/lequan/workspace/xcode/myopencv/COFW_Dataset/keypoints.txt");
	locale::global(locale("C"));
    for(int i = 0;i < img_num;i++){
        Mat_<float> temp(global_params.landmark_num,2);
        for(int j = 0;j < global_params.landmark_num;j++){
            fin>>temp(j,0); 
        }
        for(int j = 0;j < global_params.landmark_num;j++){
            fin>>temp(j,1); 
        }
        ground_truth_shapes.push_back(temp);
    }
    fin.close();
}
void LoadCofwTestData(vector<Mat_<uchar> >& images,
                      vector<Mat_<float> >& ground_truth_shapes,
                      vector<BoundingBox>& bounding_boxs){
    int img_num = 507;
    cout<<"Read images..."<<endl;
    for(int i = 0;i < img_num;i++){
        string image_name = "/Users/lequan/workspace/xcode/myopencv/COFW_Dataset/testImages/";
        image_name = image_name + to_string((_Longlong)i+1) + ".jpg";
        Mat_<uchar> temp = imread(image_name,0);
        images.push_back(temp);
    }
    
    ifstream fin;
	locale::global(locale(""));
    fin.open("/Users/lequan/workspace/xcode/myopencv/COFW_Dataset/boundingbox_test.txt");
	locale::global(locale("C"));
    for(int i = 0;i < img_num;i++){
        BoundingBox temp;
        fin>>temp.start_x>>temp.start_y>>temp.width>>temp.height;
        temp.centroid_x = temp.start_x + temp.width/2.0;
        temp.centroid_y = temp.start_y + temp.height/2.0;
        bounding_boxs.push_back(temp);
    }
    fin.close();
    locale::global(locale(""));
    fin.open("/Users/lequan/workspace/xcode/myopencv/COFW_Dataset/keypoints_test.txt");
	locale::global(locale("C"));
    for(int i = 0;i < img_num;i++){
        Mat_<float> temp(global_params.landmark_num,2);
        for(int j = 0;j < global_params.landmark_num;j++){
            fin>>temp(j,0);
        }
        for(int j = 0;j < global_params.landmark_num;j++){
            fin>>temp(j,1);
        }
        ground_truth_shapes.push_back(temp);
    }
    fin.close();
}

void LoadDataAdjust(string filepath,
              vector<Mat_<uchar> >& images,
              vector<Mat_<float> >& ground_truth_shapes,
              vector<BoundingBox> & bounding_boxs
              ){
    ifstream fin;
	locale::global(locale(""));
    fin.open(filepath);
	locale::global(locale("C"));
    string name;
    while(getline(fin,name)){
        name.erase(0, name.find_first_not_of(" \t"));
        name.erase(name.find_last_not_of(" \t") + 1);
        cout << "file:" << name <<endl;
        
        // Read Image
        Mat_<uchar> image = imread(name,0);
        
        // Read ground truth shapes
        name.replace(name.find_last_of("."), 4,".pts");
        Mat_<float> ground_truth_shape = LoadGroundTruthShape(name);
       
        // Read Bounding box
        BoundingBox bbx = CalculateBoundingBox2(ground_truth_shape);
        adjustImage2(image,ground_truth_shape,bbx);
        images.push_back(image);
        ground_truth_shapes.push_back(ground_truth_shape);    
        bounding_boxs.push_back(bbx);
    }
    fin.close();
}

void getRandomBox(const Mat_<uchar>& image, const BoundingBox& old_box, BoundingBox& new_box){
	
	int img_height=image.rows;
	int img_width=image.cols;
	
	RNG random_generator(getTickCount());
	
	int box_height=img_height;
	int box_width=img_width;
	do 
	{
		if (img_height>MINHEIGHT)
		{
			box_height= random_generator.uniform(MINHEIGHT*1.0,(double)img_height*0.8);
		}
		box_width = box_height*old_box.width/old_box.height;
	} while (box_width > img_width*0.8);

	int box_x = random_generator.uniform(0.0+0.2*img_width,(double)img_width*0.8-box_width);
	int box_y = random_generator.uniform(0.0+0.2*img_height,(double)img_height*0.8-box_height);
	new_box.start_x=box_x;
	new_box.start_y=box_y;
	new_box.width=box_width;
	new_box.height=box_height;
	new_box.centroid_x=box_x+box_width/2.0;
	new_box.centroid_y=box_y+box_height/2.0;
}

void LoadDataAdjust2(string filepath,
	vector<Mat_<uchar> >& images,
	vector<Mat_<float> >& ground_truth_shapes,
	vector<int>& ground_truth_faces,
	vector<BoundingBox> & bounding_boxs
	){
		ifstream fin;
		locale::global(locale(""));
		fin.open(filepath);
		locale::global(locale("C"));
		string name;
		while(getline(fin,name)){
			name.erase(0, name.find_first_not_of(" \t"));
			name.erase(name.find_last_not_of(" \t") + 1);
			cout << "file:" << name <<endl;

			// Read Image
			Mat_<uchar> image = imread(name,0);

			// Read ground truth shapes
			name.replace(name.find_last_of("."), 4,".pts");
			Mat_<float> ground_truth_shape = LoadGroundTruthShape(name);

			// Read Bounding box
			BoundingBox bbx = CalculateBoundingBox2(ground_truth_shape);
			adjustImage2(image,ground_truth_shape,bbx);
			images.push_back(image);
			ground_truth_shapes.push_back(ground_truth_shape);   
			ground_truth_faces.push_back(1);
			bounding_boxs.push_back(bbx);
		}
		fin.close();
}

