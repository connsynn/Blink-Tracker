#include <iostream>
#include <fstream>
#include <cv.h>
#include <highgui.h>

using namespace std;
using namespace cv; 

int main( int argc, char** argv )
{
		double distance;
		CvArr* newpair;
		newpair = cvCreateMat(1,6,CV_32FC1);
		cvSetReal1D(newpair, 0, 95);
		cvSetReal1D(newpair, 1, 4);
		cvSetReal1D(newpair, 2, 23);
		cvSetReal1D(newpair, 3, 27);
		cvSetReal1D(newpair, 4, 6);
		cvSetReal1D(newpair, 5, 5);
	
		// Input matrix size
		const int rows = 59;
		const int cols = 6;
	
		// Input matrix
	float x[rows][cols] = {{86,2,32,30,15,10},
							{86,1,24,23,5,6},
							{82,1,24,24,6,7},
							{61,2,16,16,6,6},
							{100,4,42,41,22,22},
							{101,3,40,39,21,15},
							{99,3,41,42,20,12},
							{105,3,25,20,5,5},
							{93,1,29,33,7,11},
							{101,5,44,42,22,15},
							{86,2,32,30,15,10},
							{81,1,24,25,8,9},
							{83,1,25,23,8,9},
							{84,1,26,22,9,7},
							{84,0,27,24,8,9},
							{81,1,23,24,7,8},
							{86,1,24,23,5,6},
							{80,1,20,18,6,6},
							{82,1,24,24,6,7},
							{62,2,25,16,10,9},
							{65,4,15,15,5,5},
							{69,3,22,14,5,5},
							{70,4,21,13,5,5},
							{75,4,21,9,5,5},
							{68,2,23,23,9,8},
							{74,4,22,13,5,5},
							{82,2,15,12,7,6},
							{113,0,25,20,8,5},
							{53,5,11,17,5,5},
							{87,6,16,29,8,8},
							{107,13,26,17,11,7},
							{87,6,16,29,8,8},
							{60,4,27,11,15,6},
							{73,6,20,13,4,4},
							{76,6,17,15,6,5},
							{74,6,13,15,5,4},
							{77,6,24,19,6,6},
							{73,5,22,13,4,5},
							{78,6,25,26,5,6},
							{75,6,25,19,5,5},
							{77,7,17,18,4,5},
							{81,5,22,22,8,7},
							{80,6,24,23,5,6},
							{80,5,26,22,6,7},
							{76,5,23,26,6,8},
							{79,7,22,20,6,6},
							{79,6,29,24,7,7},
							{95,1,29,29,7,6},
							{91,9,36,36,17,9},
							{88,0,20,28,4,6},
							{93,1,29,26,8,6},
							{91,0,23,29,6,7},
							{97,5,33,30,19,15},
							{96,3,28,23,12,14},
							{97,4,32,32,20,21},
							{80,2,30,29,13,10},
							{74,2,15,16,7,8},
							{76,9,13,12,4,6},
							{80,2,30,29,13,10}
		};
	
		// Place input into CvMat**
		cout << "\nEnter information in 20x6 matrix" << endl;
		CvMat** input = new CvMat*[rows];
		for(int i=0; i<rows; i++) {
			input[i] = cvCreateMat(1, cols, CV_32FC1);
			for(int j=0; j<cols; j++) {
				cvmSet(input[i], 0, j, x[i][j]);
			}
		}
		
		CvMat* output = cvCreateMat(6, 6, CV_32FC1);
		CvMat* meanvec = cvCreateMat(1, 6, CV_32FC1);
		CvMat* inversecovar  = cvCreateMat(6, 6, CV_32FC1);
		CvMat* newoutput = cvCreateMat(6, 6, CV_32FC1);	
		
		// Calculate covariance matrix
		cout << "\nCalculate covariance matrix" << endl;
		cvCalcCovarMatrix((const void **) input, rows, output, meanvec, CV_COVAR_NORMAL);
	
		// save meanvec to file
		cvSave("meanvec.xml", meanvec);
	
		// Edit Covar values to match MATLAB & Wolframalpha
		cout << "\nPrint out edited covariance matrix" << endl;
		for(int i=0; i<6; i++)
		{
			for(int j=0; j<6; j++)
			{
				cvSetReal2D(newoutput, i, j, cvGetReal2D(output,i,j) / (rows - 1));
				cout << "Edited covariance(" <<i<<","<<j<<"): ";
				printf ("%f\n", cvGetReal2D(newoutput,i,j));
				cout << "\t";
			}
			cout << endl;
		
		}
		
		// save covariance to file
		cvSave("covariance.xml", newoutput);
	
		// To invert and to apply Mahalanobis
		cvInvert( newoutput, inversecovar, CV_LU);

		distance = cvMahalanobis( meanvec, newpair, inversecovar);
		printf ("Mahalanobis: %f ", distance);

		// Clear OpenCV datastructures
		cvReleaseMat(&output);
		cvReleaseMat(&meanvec);
		
		for(int i=0; i<10; i++)
			cvReleaseMat(&input[i]);
		
		delete [] input;
		
		return 0;
}

