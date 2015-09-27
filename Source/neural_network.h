/********************************************************************************************/
/*                                                                                          */
/*   MLP-Backpropagation: A C++ implementation of Multi-Layer Perceptron (MLP)              */ 
/*                        and backpropagation algorithm for classification                  */
/*                                                                                          */
/*   N E U R A L   N E T W O R K   C L A S S   H E A D E R                                  */
/*                                                                                          */
/*   Avinash Ranganath                                                                      */
/*   Robotics Lab, Department of Systems Engineering and Automation                         */
/*   University Carlos III of Mardid(UC3M)                                                  */
/*   Madrid, Spain                                                                          */
/*   E-mail: nash911@gmail.com                                                              */
/*   https://sites.google.com/site/anashranga/                                              */
/*                                                                                          */
/********************************************************************************************/

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include<iostream>
#include<fstream>
#include<vector>
#include<string>
#include<math.h>
#include "armadillo"

#define DELTA 0.0001

using namespace std;
using namespace arma;

class NeuralNetwork{

public:
    NeuralNetwork(const unsigned int, const vector<unsigned int>, const unsigned int);

    vector<unsigned int> networkArchitecture(void) const;
    vec Layer(const unsigned int) const;
    void printTheta(void) const;
    //void setTheta(void);

    vec activation(const vec, const unsigned int);
    vec sigmoid(const vec);
    vec h(const vec);
    double cost(const mat, const mat);
    void gradientdescent(const mat, const mat);
    vector<mat> backpropagate(const mat, const mat);
    vec error(const vec, const unsigned int);


private:
    vec d_inputneurons;
    vec d_outputneurons;
    vector<vec> d_hiddenneurons;
    vector<mat> d_Theta;

    double d_alpha;
    double d_lamda;

};

#endif // NEURAL_NETWORK_H
