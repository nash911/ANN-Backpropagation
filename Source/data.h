/********************************************************************************************/
/*                                                                                          */
/*   ANN-Backpropagation: A C++ implementation of Artificial Neural Network (ANN)           */
/*                        and backpropagation algorithm for classification                  */
/*                                                                                          */
/*   D A T A   S E T   C L A S S   H E A D E R                                              */
/*                                                                                          */
/*   Avinash Ranganath                                                                      */
/*   Robotics Lab, Department of Systems Engineering and Automation                         */
/*   University Carlos III of Mardid(UC3M)                                                  */
/*   Madrid, Spain                                                                          */
/*   E-mail: nash911@gmail.com                                                              */
/*   https://sites.google.com/site/anashranga/                                              */
/*                                                                                          */
/********************************************************************************************/


#ifndef DATA_H
#define DATA_H

#include<fstream>
#include<vector>
#include<string>
#include<math.h>
#include "armadillo"

using namespace std;
using namespace arma;

class Data
{
public:
    Data(const char* fileName, const double trainPercent, const double testPercent);
    unsigned int attributeSize(const char* const) const;
    unsigned int instanceSize(const char* const) const;
    unsigned int classSize(const char* const, const unsigned int, const unsigned int) const;
    vec YClass(const char* const, const unsigned int, const unsigned int) const;

    void extractX(const char* const, const unsigned int, const unsigned int);
    vec extractY(const char* const, const unsigned int, const unsigned int) const;
    void createYMat(const char* const, const unsigned int, const unsigned int);
    void segmentDataSet(const double, const double);

    unsigned int M() const;
    unsigned int N() const;
    unsigned int K() const;
    mat X() const;
    mat Y() const;

public:
    mat d_X;
    mat d_Y;

    mat d_X_test;
    mat d_Y_test;
};

#endif // DATA_H
