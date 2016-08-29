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
	vector<Mat_<float> >& shape_residuals) {

	Mat_<float> rotation;
	float scale;
	shape_residuals.resize(bounding_boxs.size());
	for (int i = 0; i < bounding_boxs.size(); i++) {
		shape_residuals[i] = ProjectShape(ground_truth_shapes[i], bounding_boxs[i])
			- ProjectShape(current_shapes[i], bounding_boxs[i]);
		SimilarityTransform(mean_shape, ProjectShape(current_shapes[i], bounding_boxs[i]), rotation, scale);
		transpose(rotation, rotation);
		shape_residuals[i] = scale * shape_residuals[i] * rotation;
	}
}

void GetShapeResidual2(vector<int> shape_index,
					  const vector<Mat_<float> >& ground_truth_shapes,
                      const vector<Mat_<float> >& current_shapes,
                      const vector<BoundingBox>& bounding_boxs,
                      const Mat_<float>& mean_shape,
                      vector<Mat_<float> >& shape_residuals){
    Mat_<float> rotation;
    float scale;
    shape_residuals.resize(shape_index.size());
    for (int i = 0;i < shape_index.size(); i++){
        shape_residuals[i] = ProjectShape(ground_truth_shapes[shape_index[i]], bounding_boxs[shape_index[i]])
        - ProjectShape(current_shapes[shape_index[i]], bounding_boxs[shape_index[i]]);
        SimilarityTransform(mean_shape, ProjectShape(current_shapes[shape_index[i]],bounding_boxs[shape_index[i]]),rotation,scale);
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

BoundingBox CalculateBoundingBox(Mat_<uchar>& img, Mat_<float>& shape){
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
	//bbx.centroid_x = shape(15,0);
	//bbx.centroid_y = shape(15,1);
	///*circle(img, Point(bbx.centroid_x, bbx.centroid_y), 3, Scalar(255));
	//imshow("test2", img);
	//waitKey(0);*/
	///*float maxw = min(right_x - bbx.centroid_x, bbx.centroid_x - left_x);
	//float maxh = min(bottom_y - bbx.centroid_y, bbx.centroid_y - top_y);*/
 //   bbx.start_x = min(img.cols - 1.0f, max(0.f, bbx.centroid_x - (right_x - left_x)));
 //   bbx.start_y = min(img.rows - 1.0f, max(0.f, bbx.centroid_y - (bottom_y - top_y)));
	//float tmp1 = min(img.cols - 1.0f, max(0.f, bbx.centroid_x + (right_x - left_x)));
	//float tmp2 = min(img.rows - 1.0f, max(0.f, bbx.centroid_y + (bottom_y - top_y)));
 //   bbx.width  = 2 * (bbx.centroid_x - bbx.start_x);
 //   bbx.height = 2 * (bbx.centroid_y - bbx.start_y);
	bbx.start_x = left_x;
	bbx.start_y = top_y;
	bbx.height = bottom_y - top_y;
	bbx.width = right_x - left_x;
	bbx.centroid_x = bbx.start_x + bbx.width / 2.0;
	bbx.centroid_y = bbx.start_y + bbx.height / 2.0;
    return bbx;
}
void cropBoundingBox(Mat_<uchar>& img, BoundingBox box,  BoundingBox& newbox/*, Mat_<float> shape, Mat_<float>& newshape*/) {
	RNG random_generator(getTickCount());
	do{
		float step1 = random_generator.uniform(-CROP, CROP);
		float step2 = random_generator.uniform(-CROP, CROP);
		newbox.centroid_x = box.centroid_x - box.width*step1;
		newbox.centroid_y = box.centroid_y - box.height*step2;
		newbox.start_x = newbox.centroid_x - box.width/2;
		newbox.start_y = newbox.centroid_y - box.height/2;
		newbox.height = box.height;
		newbox.width = box.width;
	} while (newbox.start_x<0 || newbox.start_x>=img.cols||
			 newbox.start_y<0 || newbox.start_y>=img.rows||
		(newbox.start_x + newbox.width) < 0 || (newbox.start_x + newbox.width) >= img.cols ||
		(newbox.start_y + newbox.height)< 0 || (newbox.start_y + newbox.height) >=img.rows);
	/*float offsetx = newbox.centroid_x - box.centroid_x;
	float offsety = newbox.centroid_y - box.centroid_y;
	for (int i=0;i<28;++i)
	{
		newshape(i, 0) = shape(i, 0) + offsetx;
		newshape(i, 1) = shape(i, 1) + offsety;
	}*/
}
void adjustImage(Mat_<uchar>& img,
                 Mat_<float>& ground_truth_shape,
                 BoundingBox& bounding_box){
	/*imshow("test",img);
	waitKey(0);*/
    float left_x  = max(1.0, (double)bounding_box.centroid_x - bounding_box.width*0.8);
    float top_y   = max(1.0, (double)bounding_box.centroid_y - bounding_box.height*0.8);
    float right_x = min(img.cols-1.0,(double)bounding_box.centroid_x+bounding_box.width*0.8);
    float bottom_y= min(img.rows-1.0,(double)bounding_box.centroid_y+bounding_box.height*0.8);
    
	img = img.rowRange((int)top_y,(int)bottom_y).colRange((int)left_x,(int)right_x).clone();
	

	

	bounding_box.start_x = ((int)right_x - (int)left_x)*CROP;
	bounding_box.start_y = ((int)bottom_y - (int)top_y)*CROP;
	bounding_box.width = ((int)right_x - (int)left_x)*(1-2*CROP)-1;
	bounding_box.height = ((int)bottom_y - (int)top_y)*(1-2*CROP)-1;
	bounding_box.centroid_x = bounding_box.start_x + bounding_box.width / 2.0;
	bounding_box.centroid_y = bounding_box.start_y + bounding_box.height / 2.0;
    
    for(int i=0;i<ground_truth_shape.rows;i++){
        ground_truth_shape(i,0) = ground_truth_shape(i,0)-left_x;
        ground_truth_shape(i,1) = ground_truth_shape(i,1)-top_y;
    }
	//imshow("test1",img);
	//waitKey(0);
	float ori_height=img.rows;
	float ori_weight=img.cols;

	if (ori_height>MAXHEIGHT_POS)
	{
		float scale=MAXHEIGHT_POS/ori_height;
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

void getRandomBox(const Mat_<uchar>& image, const BoundingBox& old_box, BoundingBox& new_box){
	RNG random_generator(getTickCount());
	do{
		new_box.start_x = random_generator.uniform(0, image.cols - 1);
		new_box.start_y = random_generator.uniform(0, image.rows - 1);
		new_box.height = random_generator.uniform(MINHEIGHT, image.rows - 1);
		new_box.width = (int)(new_box.height*old_box.width / old_box.height);
	} while (new_box.start_x+ new_box.width>=image.cols||
		new_box.start_y + new_box.height>=image.rows);
	
	new_box.centroid_x = new_box.start_x + new_box.width /2.0;
	new_box.centroid_y = new_box.start_y + new_box.height /2.0;
}

float CalculateError(const Mat_<float>& ground_truth_shape, const Mat_<float>& predicted_shape) {
	Mat_<float> temp;
	//temp = ground_truth_shape.rowRange(36, 37)-ground_truth_shape.rowRange(45, 46);
	temp = ground_truth_shape.rowRange(1, 2) - ground_truth_shape.rowRange(6, 7);
	float x = mean(temp.col(0))[0];
	float y = mean(temp.col(1))[1];
	float interocular_distance = sqrt(x*x + y*y);
	float sum = 0;
	for (int i = 0; i < ground_truth_shape.rows; i++) {
		sum += norm(ground_truth_shape.row(i) - predicted_shape.row(i));
	}
	return sum / (ground_truth_shape.rows*interocular_distance);
}
