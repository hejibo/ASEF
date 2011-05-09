#include "face_sense.h"
#include <assert.h>
#include <math.h>

#include <dirent.h>


#define MY_IMG_WIDTH 256
#define MY_IMG_WIDTH 256
#define MY_FACE_RESIZE 128

int main( int argc, char** argv )
{
	
	char window_name[] = "face_tracker";
	
	CvSize frame_size = cvSize(MY_IMG_WIDTH, IMG_HEIGHT);
	
	IplImage* frame = NULL;
	IplImage* clr_img = cvCreateImage( cvSize(frame_size.width,frame_size.height), IPL_DEPTH_8U, 3 );
	IplImage* cur_img = cvCreateImage(cvGetSize(clr_img), IPL_DEPTH_8U, 1);
  IplImage* front_face = cvCreateImage(cvSize(MY_FACE_RESIZE, MY_FACE_RESIZE), IPL_DEPTH_8U, 1); 
  IplImage* front_face_flip = cvCreateImage(cvSize(MY_FACE_RESIZE, MY_FACE_RESIZE), IPL_DEPTH_8U, 1); 

  CvMat* frontfacemat = cvCreateMat(MY_FACE_RESIZE, MY_FACE_RESIZE, CV_32FC1);
  CvMat* anyfacemat = cvCreateMat(MY_FACE_RESIZE, MY_FACE_RESIZE, CV_32FC1);

	CvRect face_rect = cvRect(0,0,0,0);
  double front_symmetry;	
  double face_symmetry;
	int flag_confirm = 0;
  
  CvFont font;
  cvInitFont(&font, CV_FONT_HERSHEY_COMPLEX, 1.0, 1.0, 0.0, 2, CV_AA);

	printf("allocation done\n");
	
	int isFoundFace = 0;
	
	cvNamedWindow(window_name);
	CvCapture* capture = cvCaptureFromCAM(-1);
	if( !capture ) {
		fprintf( stderr, "ERROR: capture is NULL \n" );
		return -1;
	}
	
	CvHaarClassifierCascade* cascade = load_object_detector(argv[1]);	
	cvNamedWindow(window_name, 0 );
	//cvNamedWindow("test", 0 );

	int iframe = 0;
	
	// ***************************************allocation for eyes **********************************
	//直接把要处理图像的大小以及定义好的左眼和右眼的区域写入

  FILE *feyefilterb = fopen("./eyefilterb","rb");
  int col=128;
  int row =128;
  CvRect lrect = cvRect(23, 35, 32, 32);
  CvRect rrect = cvRect(71, 32, 32, 32);

  float* lf = (float*)malloc(4*col*row);
  float* rf = (float*)malloc(4*col*row);
   //get left eye filter
  for(int i=0; i<128*128; i++){
    fscanf(feyefilterb, "%f\n", lf+i);
		
      }
	  //get right eye filter
	for(int i=0; i<128*128; i++){
		fscanf(feyefilterb, "%f\n", rf+i);
		
  }
	fclose(feyefilterb);

  //

  CvMat* lfilter = cvCreateMatHeader(row, col, CV_32FC1);
  CvMat* rfilter = cvCreateMatHeader(row, col, CV_32FC1);
  
  cvSetData(lfilter, lf, CV_AUTO_STEP);
  cvSetData(rfilter, rf, CV_AUTO_STEP);

  CvScalar avg, std;
  cvAvgSdv(lfilter, &avg, &std);

  cvConvertScale(lfilter, lfilter, 1.0/std.val[0], (-avg.val[0]*1.0)/std.val[0]);
  printf("filter avg, std: left: avg%f, std%f\n", std.val[0], avg.val[0]);
  cvAvgSdv(rfilter, &avg, &std);
  cvConvertScale(rfilter, rfilter, 1.0/std.val[0], (-avg.val[0]*1.0)/std.val[0]);
  printf("filter avg, std: righ: avg%f, std%f\n", std.val[0], avg.val[0]);
//  cvShowImage("test", lfilter); 
  CvMat* lfilter_dft = cvCreateMat(row, col, CV_32FC1);
  CvMat* rfilter_dft = cvCreateMat(row, col, CV_32FC1);
	
  CvMat* image = cvCreateMat(row, col, CV_32FC1);
  CvMat* image_tile = cvCreateMat(row, col, CV_8UC1);
  CvMat* lcorr = cvCreateMat(row, col, CV_32FC1);
  CvMat* rcorr = cvCreateMat(row, col, CV_32FC1);
	
  CvMat* lroi = cvCreateMatHeader(row, col, CV_32FC1);
  CvMat* rroi = cvCreateMatHeader(row, col, CV_32FC1);
	
  cvDFT(lfilter, lfilter_dft, CV_DXT_FORWARD);
  cvDFT(rfilter, rfilter_dft, CV_DXT_FORWARD);
	
  cvGetSubRect(lcorr, lroi, lrect);
  cvGetSubRect(rcorr, rroi, rrect);
	
  CvMat* lut = cvCreateMat(256, 1, CV_32FC1);
  for (int i = 0; i<256; i++){
    cvmSet(lut, i, 0, 1.0 + i);
  }
  cvLog(lut, lut);
  
	//*********************************allocation for eye done*******************************************
	
	char pathname[] = "/Users/xy/coding/face/img/JAFFE/";
	DIR *dirp = opendir(pathname);
	struct dirent* dirstruct;
	
	char filename[1000];
	//Detection
	while((dirstruct = readdir(dirp))!=NULL){
		if(  strcmp( dirstruct->d_name + strlen(dirstruct->d_name) - 5, ".tiff") !=0){
			continue;
		}
		strncpy(filename, pathname, 1000);
		strncat(filename, dirstruct->d_name, 1000-strlen(pathname));
		printf("%s\t", dirstruct->d_name); 
		
		cur_img = cvLoadImage(filename);
		
		iframe ++;
		
		
    isFoundFace = !detect_objects(cur_img , cascade, &face_rect);

    if( isFoundFace ){
      CvMat* face_clr_img = cvCreateMatHeader(face_rect.height,face_rect.width, CV_8UC3); 
      CvMat* face_gry_img = cvCreateMat(face_rect.height,face_rect.width, CV_8UC1); 

      cvGetSubRect(cur_img, face_clr_img, face_rect);
      cvCvtColor(face_clr_img, face_gry_img, CV_RGB2GRAY);
      
      //*************find eye****************

      double xscale = double(image_tile->width)/double(face_gry_img->width);
      double yscale = double(image_tile->height)/double(face_gry_img->height);

      cvResize(face_gry_img, image_tile);

      cvLUT(image_tile, image, lut);

      cvDFT(image, image, CV_DXT_FORWARD);
      cvMulSpectrums(image, lfilter_dft, lcorr, CV_DXT_MUL_CONJ);
      cvMulSpectrums(image, rfilter_dft, rcorr, CV_DXT_MUL_CONJ);
      
      cvDFT(lcorr, lcorr, CV_DXT_INV_SCALE);
      cvDFT(rcorr, rcorr, CV_DXT_INV_SCALE);

      CvPoint leye = cvPoint(0,0);
      CvPoint reye = cvPoint(0,0);
      CvPoint lMinLoc = cvPoint(0,0);
      CvPoint rMinLoc = cvPoint(0,0);
      double lminVal, lmaxVal, rminVal, rmaxVal;
      cvMinMaxLoc(lroi, NULL, NULL, NULL, &leye);
      cvMinMaxLoc(rroi, NULL, NULL, NULL, &reye);
   
      leye.x = (lrect.x + leye.x)/xscale + face_rect.x;
      leye.y = (lrect.y + leye.y)/yscale + face_rect.y;
      reye.x = (rrect.x + reye.x)/xscale + face_rect.x;
      reye.y = (rrect.y + reye.y)/yscale + face_rect.y;


     // printf("left eye: %d, %d \tright eye: %d, %d\n", lrect.x, lrect.y,rrect.x,rrect.y);


      cvCircle(cur_img, leye, 3, CV_RGB(255,0,0), 3, CV_AA);
      cvCircle(cur_img, reye, 3, CV_RGB(255,0,0), 3, CV_AA);		
      
      
      //*************find eye done***********


      cvResize(face_gry_img, front_face);
      cvEqualizeHist(front_face, front_face);//
      cvSmooth(front_face, front_face, CV_GAUSSIAN, 5);

      cvConvert(front_face, frontfacemat);
      CvScalar avg, std;
      cvAvgSdv(frontfacemat, &avg, &std);
      cvAddS(frontfacemat, cvScalar(-avg.val[0]), frontfacemat);
      cvScale(frontfacemat, frontfacemat, 1.0/std.val[0]);

      cvFlip(front_face, front_face_flip,1);
      front_symmetry = cvNorm(front_face, front_face_flip);

      cvReleaseMatHeader(&face_clr_img);
      cvReleaseMat(&face_gry_img);

      cvRectangle( cur_img, cvPoint(face_rect.x,face_rect.y), cvPoint(face_rect.x+face_rect.width, face_rect.y+face_rect.height), CV_RGB(255,0,0), 3 );
			
			printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n", face_rect.x,face_rect.y, face_rect.width,face_rect.height, leye.x, leye.y, reye.x , reye.y );
			
    }
    cvFlip(cur_img, cur_img, 1);
    cvShowImage( window_name, cur_img);
		cvWaitKey(80);
    //if(isFoundFace)  break;
	} 	

	//***********************clean for eye*********************
	
  cvReleaseMatHeader(&lfilter);
  cvReleaseMatHeader(&rfilter);
  cvReleaseMat(&lfilter_dft);
  cvReleaseMat(&rfilter_dft);
  cvReleaseMat(&image);
	cvReleaseMat(&image_tile);
  cvReleaseMat(&lcorr);
  cvReleaseMat(&rcorr);
  cvReleaseMat(&lroi);
  cvReleaseMat(&rroi);
  cvReleaseMat(&lut);
  
  free(lf);
  free(rf);
	//***********************clean for eye done *******************
		
	printf("\n");	
	cvReleaseCapture( &capture);
	cvDestroyWindow(window_name);
	cvReleaseHaarClassifierCascade( &cascade );
	cvReleaseImage( &cur_img );
  cvReleaseImage(&front_face);
  cvReleaseImage(&front_face_flip);
  cvReleaseMat(&anyfacemat);
  cvReleaseMat(&frontfacemat);
	return 0;
}
