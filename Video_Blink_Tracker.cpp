#include <stdio.h>
#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <fstream>

//Function declarations
char pointIsInsideContour(CvSeq *contour, int x, int y); //Returns whether the given position is inside a contour

IplImage *img1; // first image taken
int Sx=0,Sy=0,Wr=0,Wl=0,Hr=0,Hl=0; // vector properties
CvRect r1,r2,eye; // contours


int main (int argc, const char * argv[]) {
	
	// exit main program loop?
	char quit = 0; 
	char grab_frame = 1;
	int nc = 0; // connected components
	double distance; // mahal distance
	CvFont font; // font for put text
	CvArr *newpair; // new data for comparison
	int previous_frame=0; // calc frame
	int same_blink=0; // if the blink occurred too closely
	time_t begin, end; // calc processing time
	
	// allocated memory
	CvMemStorage *storage = cvCreateMemStorage(0);
	CvSeq *contours = 0;
	
	// get a video
	CvCapture *capture = cvCaptureFromAVI("Movie Recording 27.mov");
	
	// start timestamp
	time(&begin);
	
	// get the video image size
	IplImage *imgsize;
	//imgsize = cvQueryFrame( camera );
	imgsize = cvQueryFrame( capture );
	if( !imgsize ) return -1;
	
	// get a buffer image that is the same size as img1 and 2 
	IplImage *imggray1 = cvCreateImage( cvGetSize( imgsize ), IPL_DEPTH_8U, 1);
	IplImage *imggray2 = cvCreateImage( cvGetSize( imgsize ), IPL_DEPTH_8U, 1);
	IplImage *imggray3 = cvCreateImage( cvGetSize( imgsize ), IPL_DEPTH_8U, 1);
	IplImage *imgcross = cvCreateImage( cvGetSize( imgsize ), IPL_DEPTH_8U, 1);
	
	// create a window
	cvNamedWindow("Video", 0); 
	
	// define kernal for image smoothing
	IplConvKernel *kernel = cvCreateStructuringElementEx(3, 3, 1, 1, CV_SHAPE_CROSS, NULL);
	
	// initialize blink count
	int blinkingCount = 0;
	int c = cvWaitKey(30);
	// exit the loop when the user sets quit to true by pressing the "esc" key
	while(!quit){
		int c = cvWaitKey(30); //Wait 30 ms for the user to press a key.
		
		//Respond to key pressed.
		switch(c){
			case 32: //Space
				grab_frame = !grab_frame; //Reverse the value of grab_frame. That way, the user can toggle by pressing the space bar.
				break;
			case 27: //Esc: quit application when user presses the 'esc' key.
				quit = 1; //Get out of loop
				break;
		};
		
		//If we don't have to grab a frame, we're done for this pass.
		if(!grab_frame)continue;
		
		//Grab 1st frame 
		img1 = cvQueryFrame(capture);
		//if(!img1)continue; 
		if(!img1)break;
		
		// convert from RGB to greyscale 
		cvCvtColor(img1, imggray1, CV_RGB2GRAY);
		
		// get frame property
		int current_frame  = cvGetCaptureProperty(capture, CV_CAP_PROP_POS_FRAMES);
		
		//Grab 2nd frame
		IplImage *img2 = cvQueryFrame(capture);
		if(!img2)continue; 		
		
		// convert from RGB to greyscale before processing further.
		cvCvtColor(img2, imggray2, CV_RGB2GRAY);
		
		// compute difference
		cvAbsDiff( imggray1, imggray2, imggray3 );
		
		// convert to bw
		IplImage* imgbw = cvCreateImage(cvGetSize(imggray3),IPL_DEPTH_8U,1);		
		cvThreshold(imggray3, imgbw, 16, 255, CV_THRESH_BINARY);
		
		// apply convolution kernal
		cvMorphologyEx( imgbw, imgcross , NULL , kernel, CV_MOP_OPEN, 1);
		
		// find connected components
		nc = cvFindContours(imgcross, storage, &contours, sizeof(CvContour), CV_RETR_TREE, CV_CHAIN_APPROX_NONE, cvPoint(0,0));
		
		// Create a clone for contour coloring, or else contours will be 0 at the end
		CvSeq *contours_replica = contours;
		// loop through all contours found
		for( ; contours != 0; contours = contours->h_next )
        {
			// initialize new mahal distance
			double newdistance=10000;
			
			// Loop thru 1st contour
			r1 = cvBoundingRect(contours, 1);
			
			// get the centroid of the contour
			CvPoint point = cvPoint(
									r1.x + (r1.width / 2),
									r1.y + (r1.height / 2)
									);
			for( ; contours_replica != 0; contours_replica = contours_replica->h_next )
			{
				// draw contours area
				CvScalar color = CV_RGB( rand()&255, rand()&255, rand()&255 );
				cvDrawContours( img1, contours_replica, color, color, -1, CV_FILLED, 8 );
				
				// loop thru 2nd contour
				r2 = cvBoundingRect(contours_replica, 1);
				
				// get the centroid of the contour
				CvPoint point2 = cvPoint(
										r2.x + (r2.width / 2),
										r2.y + (r2.height / 2)
										);
				
				// if centroid located in another contour
				if(pointIsInsideContour(contours, (r2.x + (r2.width / 2)),(r2.y + (r2.height / 2)))) break;
				
				// reasonable horizontal distance, based on the components' width
				int dist_ratio = abs(r1.x - r2.x) / r1.width;
				if (dist_ratio < 2 || dist_ratio > 5)
				break;
				
				//creating vector properties
				Sx = abs(r1.x - r2.x);
				Sy = abs(r1.y - r2.y);
				Wr = r1.width;
				Wl = r2.width;
				Hr = r1.height;
				Hl = r2.height;
				
				// assiging vec properties into Array
				newpair = cvCreateMat(1,6,CV_32FC1);
				cvSetReal1D(newpair, 0, Sx);
				cvSetReal1D(newpair, 1, Sy);
				cvSetReal1D(newpair, 2, Wr);
				cvSetReal1D(newpair, 3, Wl);
				cvSetReal1D(newpair, 4, Hr);
				cvSetReal1D(newpair, 5, Hl);
				
				// read from pre-processed xml file (from C++ Mahal Training)
				CvMat* trained_matrix;
				trained_matrix = (CvMat*)cvLoad("covariance.xml");
				CvMat* meanvec;
				meanvec = (CvMat*)cvLoad("meanvec.xml");
				CvMat*inversecovar  = cvCreateMat(6, 6, CV_32FC1);
				
				// invert and to apply Mahalanobis
				cvInvert( trained_matrix, inversecovar, CV_LU);
				
				// calc mahalanobis distance
				distance = cvMahalanobis( meanvec, newpair, inversecovar);
								
				// to make sure we take the highest posibility of eye pairs in each image
				if (newdistance > distance) {
					newdistance	= distance;
					eye = r2;
				}
				
				// assume it's the same blink if it happened too fast for human being
				if (current_frame - previous_frame <= 10000)
				{
					same_blink = 1;
				}
			}
			
			// consider new blink if connected components >= 2, mahal distance <= 5, not a same blink in sequence
			if (nc >= 2 && newdistance <= 5 && same_blink == 0)
			{
				cvRectangle(img1, cvPoint(r1.x, r1.y), cvPoint((r1.x + r1.width),(r1.y + r1.height)), CV_RGB(255,0,0),  1, 8, 0);
				cvRectangle(img1, cvPoint(eye.x, eye.y), cvPoint((eye.x + eye.width),(eye.y + eye.height)), CV_RGB(255,0,0),  1, 8, 0);					cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1.0, 1.0, 0, 1, CV_AA);
				cvPutText(img1, "Eye pair!", cvPoint(10, 130), &font, cvScalar(255, 255, 255, 0));
				cvShowImage("Video", img1);
				blinkingCount++;
				previous_frame = current_frame;
				printf("blinking count : %d\n",blinkingCount);
				break;
			}
			else {
				// display the image in the window.
				cvShowImage("Video", img1);
			}
			//reset counter for same_blink
			same_blink = 0;
        }
	}
	
	// clean up before quitting.
	cvDestroyAllWindows(); 
	
	// release the camera	
	cvReleaseCapture(&capture);
	
	// release memory
	cvReleaseMemStorage(&storage);
	
	// release images
	cvReleaseImage(&imggray1);
	cvReleaseImage(&imggray2);
	cvReleaseImage(&imggray3);
	cvReleaseImage(&imgcross);
	
	// end timestamp
	time(&end);
	cout << "Time elapsed: " << difftime(end, begin) << " seconds"<< endl;

	return 0;
}

// returns whether one contour point is inside the given contour.
char pointIsInsideContour(CvSeq *contour, int x, int y){
	//We know that a point is inside a contour if we can find one point in the contour that is immediately to the left,
	//one that is immediately on top, one that is immediately to the right and one that is immediately below the (x,y) point.
	//We will update the boolean variables below when these are found:
	char found_left=0, found_top=0, found_right=0, found_bottom=0;
	int count, i; 
	CvPoint *contourPoint; 	
	// do not proceed if there is no contour
	if(!contour)return 0;
	
	count = contour->total; //The total field holds the number of points in the contour.
	
	for(i=0;i<count;i++){ //So, for every point in the contour...
		//We retrieve a pointer to the point at index i using the useful macro CV_GET_SEQ_ELEM
		contourPoint = (CvPoint *)CV_GET_SEQ_ELEM(CvPoint,contour,i);
		
		if(contourPoint->x == x){ //If the point is on the same vertical plane as (x,y)...
			if(contourPoint->y < y)found_top = 1; //and is above (x,y), we found the top.
			else found_bottom = 1; //Otherwise, it's the bottom.
		}
		if(contourPoint->y == y){ //Do the same thing for the horizontal axis...
			if(contourPoint->x < x)found_left = 1;
			else found_right = 1;
		}
	}
	
	return found_left && found_top && found_right && found_bottom; //Did we find all four points?
}

