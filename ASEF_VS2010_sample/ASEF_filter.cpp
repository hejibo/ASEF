// ASEF_filter.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include<fstream>
using namespace std; 
int main( int argc, char** argv )
{
	IplImage *face=cvLoadImage("face.jpg",0);//载入灰度图像
	
	CvMat * dft_face=cvCreateMat(face->height,face->width, CV_32FC2);//双通道DFT频域
	CvMat * Real_part=cvCreateMat(face->height,face->width, CV_32FC1);;//单通道DFT频域
	CvMat * Imagary_part=cvCreateMat(face->height,face->width, CV_32FC1);;//单通道DFT频域
	
	// store the right and left filter data
	CvMat* lfilter= cvCreateMatHeader(face->height, face->width, CV_32FC1);
    CvMat* rfilter= cvCreateMatHeader(face->height, face->width, CV_32FC1);
	
	CvMat* lfilter_complex= cvCreateMat(face->height, face->width, CV_32FC2);
    CvMat* rfilter_complex= cvCreateMat(face->height, face->width, CV_32FC2);

	
	 // define the region of interest mat
	 CvMat* lroi = cvCreateMatHeader(32, 32, CV_32FC1);
     CvMat* rroi = cvCreateMatHeader(32, 32, CV_32FC1);

	 CvRect lfilter_rect = cvRect(23, 35, 32, 32);
	 CvRect rfilter_rect = cvRect(71, 32, 32, 32);


	CvMat* lut = cvCreateMat(256, 1, CV_32FC1);
	CvMat* face_lut = cvCreateMat(face->height, face->width, CV_32FC1);
	
	for (int i = 0; i<256; i++){
		cvmSet(lut, i, 0, 1.0 + i);
	  }
	cvLog(lut, lut);//归一化face
	cvLUT(face, face_lut, lut);

	cvCopy( face_lut,Real_part, NULL );
	cvZero(Imagary_part);
	cvMerge(Real_part, Imagary_part, NULL, NULL, dft_face);
	cvDFT(dft_face, dft_face, CV_DXT_FORWARD);

	//read right filter from txt file
	float *lfilter_txt=new float[face->height*face->width];
    float *rfilter_txt=new float[face->height*face->width];

	//read left filter data
	ifstream infile;
    infile.open("left_filter.txt");
	if(!infile.good())
	{
	  exit(1);
	}
	int i=0;
	while(!infile.eof())
	{
		infile>>*(lfilter_txt+i);
	    i++;
	}
	infile.close();
	//read right filter data
	infile.open("right_filter.txt");
	if(!infile.good())
	{
	  exit(1);
	}
	i=0;
	while(!infile.eof())
	{
		infile>>*(rfilter_txt+i);
	    i++;
	}
	infile.close();

	cvSetData(lfilter, lfilter_txt, CV_AUTO_STEP);
	cvSetData(rfilter, rfilter_txt, CV_AUTO_STEP);
	
	
	
	CvPoint leye = cvPoint(0,0);
    CvPoint reye = cvPoint(0,0);

	cvCopy( lfilter,Real_part, NULL );
	cvZero(Imagary_part);
	cvMerge(Real_part, Imagary_part, NULL, NULL, lfilter_complex);
    cvDFT(lfilter_complex, lfilter_complex, CV_DXT_FORWARD);
	cvMulSpectrums(dft_face,lfilter_complex,lfilter_complex,CV_DXT_FORWARD);
	cvDFT(lfilter_complex, lfilter_complex, CV_DXT_INV_SCALE);
	cvSplit(lfilter_complex, Real_part, Imagary_part, 0, 0 );
	cvGetSubRect(Real_part, lroi, lfilter_rect);
	cvMinMaxLoc(lroi, NULL, NULL, NULL, &leye);

	

	cvCopy( rfilter,Real_part, NULL );
	cvZero(Imagary_part);
	cvMerge(Real_part, Imagary_part, NULL, NULL, rfilter_complex);
    cvDFT(rfilter_complex, rfilter_complex, CV_DXT_FORWARD);
	cvMulSpectrums(dft_face,rfilter_complex,rfilter_complex,CV_DXT_FORWARD);
	cvDFT(rfilter_complex, rfilter_complex, CV_DXT_INV_SCALE);
	cvSplit(rfilter_complex, Real_part, Imagary_part, 0, 0 );
	cvGetSubRect(Real_part, rroi, rfilter_rect);
	cvMinMaxLoc(rroi, NULL, NULL, NULL, &reye);



    //set the region of interest
  

	cout<<"left_x"<<(leye.x+23);
	cout<<"left_y"<<(leye.y+35);

	cout<<"right_x"<<(reye.x+71);
	cout<<"right_y"<<(leye.y+35);
    cvReleaseImage( &face );
		
   

	return 0;
}


