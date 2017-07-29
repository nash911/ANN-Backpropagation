/********************************************************************************************/
/*                                                                                          */
/*   ANN-Backpropagation: A C++ implementation of Artificial Neural Network (ANN)           */
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

#define ALPHA 0.3
#define LAMDA 0.0
#define LEARNING_CURVE_DELTA 0.0001

using namespace std;
using namespace arma;

class NeuralNetwork{

public:
    NeuralNetwork(const unsigned int&, const vector<unsigned int>&, const unsigned int&);

    void set_alpha(const double&);
    void set_lamda(const double&);
    double alpha(void) const;
    double lamda(void) const;

    vector<unsigned int> networkArchitecture(void) const;
    vec Layer(const unsigned int) const;
    void printTheta(void) const;

    vec activation(const vec&, const unsigned int&);
    vec sigmoid(const vec&) const;
    vec h_Theta(const vec&);
    double cost(const vector<mat>&, const mat&, const mat&);
    void train(const mat&, const mat&);
    void gradientdescent(vector<mat>&, const vector<mat>&);
    vector<mat> backpropagate(const vector<mat>&, const mat&, const mat&);
    vec error(const mat&, const vec&, const unsigned int&) const;

    void gradientCheck(const vector<mat>&, const vector<mat>&, const mat&, const mat&);
    vector<mat> numericalGradient(const vector<mat>&, const mat&, const mat&);


private:
    vec d_inputneurons;
    vec d_outputneurons;
    vector<vec> d_hiddenneurons;
    vector<mat> d_Theta;

    double d_alpha;
    double d_lamda;

};

#endif // NEURAL_NETWORK_H
