/********************************************************************************************/
/*                                                                                          */
/*   ANN-Backpropagation: A C++ implementation of Artificial Neural Network (ANN)           */
/*                        and backpropagation algorithm for classification                  */
/*                                                                                          */
/*   N E U R A L   N E T W O R K   C L A S S                                                */
/*                                                                                          */
/*   Avinash Ranganath                                                                      */
/*   Robotics Lab, Department of Systems Engineering and Automation                         */
/*   University Carlos III of Mardid(UC3M)                                                  */
/*   Madrid, Spain                                                                          */
/*   E-mail: nash911@gmail.com                                                              */
/*   https://sites.google.com/site/anashranga/                                              */
/*                                                                                          */
/********************************************************************************************/

#include "neural_network.h"

// CONSTRUCTOR

/// Constructs a neural network architecture.
/// Initializes learning rate parameter 𝛼 and regularization parameter λ to defaults.
/// Creates and randomly initializes matrices Θ⁽l⁾ ∀ l = 1,2,...,(L-1).
/// @param inputNeurons Number of neurons in the input layer.
/// @param hiddenLayers Number of hidded laayers, and hidden neurons in each hidden layer.
/// @param outputNeurons Number of neurons in the output layer.

NeuralNetwork::NeuralNetwork(const unsigned int inputNeurons, const vector<unsigned int> hiddenLayers, const unsigned int outputNeurons)
{
    d_alpha = ALPHA;
    d_lamda = LAMDA;

    d_inputneurons.set_size(inputNeurons);
    d_outputneurons.set_size(outputNeurons);

    for(unsigned int hl = 0; hl < hiddenLayers.size(); hl++)
    {
        vec hidLayer(hiddenLayers[hl]);
        d_hiddenneurons.push_back(hidLayer);
    }

    unsigned int rows = 0;
    unsigned int cols = 0;
    unsigned int layers = networkArchitecture().size();

    for(unsigned int l = 0; l < (layers - 1); l++)
    {
        rows = networkArchitecture()[l+1];
        cols = networkArchitecture()[l]+1;

        cout << endl << "Layer " << l+1 <<  ":    Rows: " << rows << "    Cols: " << cols << endl;

        //--Random initialization of Theta matrix Θ⁽l⁾--//
        mat theta(rows, cols, fill::randu);
        d_Theta.push_back(theta);
    }
}


void NeuralNetwork::set_alpha(const double alpha)
{
    if(alpha < 0.0)
    {
        cerr << "ANN-Backpropagation Error: NeuralNetwork class." << endl
             << "void set_alpha(const double) methon." << endl
             << "Parameter alpha: " << alpha << " must be >= 0."
             << endl;

        exit(1);
    }

    d_alpha = alpha;
}


void NeuralNetwork::set_lamda(const double lamda)
{
    if(lamda < 0.0)
    {
        cerr << "ANN-Backpropagation Error: NeuralNetwork class." << endl
             << "void set_lamda(const double) methon." << endl
             << "Parameter lamda: " << lamda << " must be >= 0."
             << endl;

        exit(1);
    }

    d_lamda = lamda;
}


double NeuralNetwork::alpha(void) const
{
    return d_alpha;
}


double NeuralNetwork::lamda(void) const
{
    return d_lamda;
}


// vector<unsigned int> networkArchitecture(void) const

/// Returns a vector containing network architecture size.
/// Number of layers in the network is the size of the vector.
/// Number of neurons in each layer is stored in each row of the vector.

vector<unsigned int> NeuralNetwork::networkArchitecture(void) const
{
    vector<unsigned int> architecture;

    architecture.push_back(d_inputneurons.n_rows);
    for(unsigned int hl=0; hl<d_hiddenneurons.size(); hl++)
    {
        architecture.push_back(d_hiddenneurons[hl].n_rows);
    }
    architecture.push_back(d_outputneurons.n_rows);

    return architecture;
}


// vector<unsigned int> networkArchitecture(void) const

/// Returns a vector of activations a⁽l⁾, where l=1,2,...,L, of layer l.
/// The activations are from the most recent forward propagation of the network.
/// @param l Layer of the network.

vec NeuralNetwork::Layer(const unsigned int l) const
{
    if(l >= networkArchitecture().size())
    {
        cerr << "ANN-Backpropagation Error: NeuralNetwork class." << endl
             << "vec Layer(const unsigned int) methon." << endl
             << "Layer l: " << l << " should be < network architecture size: " << networkArchitecture().size()
             << endl;

        exit(1);
    }

    vec a;
    vec a_0 = ones<vec>(1);

    if(l == 0)
    {
        a = d_inputneurons;

        //--Adding element a_0 ∀ l≠L--//
        a.insert_rows(0, a_0);
    }
    else if(l == networkArchitecture().size() - 1)
    {
        a = d_outputneurons;
    }
    else
    {
        a = d_hiddenneurons[l-1];

        //--Adding element a_0 ∀ l≠L--//
        a.insert_rows(0, a_0);
    }

    return a;
}


// void printTheta(void) const

/// Prints on screen Θ⁽l⁾ ∀ l=1,2,...,(L-1).

void NeuralNetwork::printTheta(void) const
{
    for(unsigned int l=0; l<d_Theta.size(); l++)
    {
        cout << endl << "Theta_" << l+1 << ":" << endl;
        d_Theta[l].print();
    }
}


// vec activation(const vec, const unsigned int)

/// Calculates activation a⁽l⁾ = g(z), for a given layer l ∈ 1,2,...,L.
/// Stores vector a⁽l⁾ in layer l of the network.
/// Returns a vector of activations a⁽l⁾ for layer l.
/// @param inputVec Vector v of inputs to calculate activation. v = x_i where l=1, or v = a⁽l-1⁾ where l=2,3,...,L
/// @param l Layer of the network.

vec NeuralNetwork::activation(const vec inputVec, const unsigned int l)
{
    if(l >= networkArchitecture().size())
    {
        cerr << "ANN-Backpropagation Error: NeuralNetwork class." << endl
             << "vec activation(const vec, const unsigned int) methon." << endl
             << "Layer l: " << l << " should be < network architecture size: " << networkArchitecture().size()
             << endl;

        exit(1);
    }

    //--a⁽l⁾ = x_i ∀ l=1--//
    vec a = inputVec;

    //--If l=2,3,4,...,L--//
    if(l > 0)
    {
        //--g(Θ⁽l⁾ * a⁽l-1⁾)--//
        a = sigmoid(d_Theta[l-1] * a);
    }

    //--Storing a⁽l⁾ in layer l of the neural network--//
    if(l == 0)
    {
        d_inputneurons = a;
    }
    else if(l < networkArchitecture().size() - 1)
    {
        d_hiddenneurons[l-1] = a;
    }
    else
    {
        d_outputneurons = a;
    }


    //--Adding bias neuron a⁽l⁾_0--//
    if(l < (networkArchitecture().size() - 1))
    {
        //--Adding element a_0 ∀ l≠L--//
        vec a_0 = ones(1);
        a.insert_rows(0, a_0);
    }

    return a;
}


// vec sigmoid(const vec)

/// Calculates sigmoid for a given vector.
/// Returns a vector of sigmoids z⁽l⁾ for layer l.
/// @param z Input vector z = Θ⁽l⁾ * a⁽l-1⁾ where l=2,3,...,L

vec NeuralNetwork::sigmoid(const vec z)
{
    vec one = ones<vec>(z.n_rows);

    //--g(z) = ¹/(1 + e^-z)--//
    return(one/(one + exp(-z)));

}


// vec h_Theta(const vec)

/// Calculates hΘ(x) = g(z⁽L⁾)
/// Returns a vector v ∈ R^k, predicting one instance of data x_i.
/// @param x Input vector x, which is a feature vector in the data set.

vec NeuralNetwork::h_Theta(const vec x)
{
    vec a = x;
    unsigned int layers = networkArchitecture().size();

    for(unsigned int l=0; l<layers; l++)
    {
        a = activation(a, l);
    }

    return(a);
}


// double cost(const vector<mat>, const mat, const mat)

/// Calculates and returns the regularized cost of the neural network, given Θ, input matric X and k-class label matric Y.
/// @param Theta A vector of Theta matrices Θ⁽l⁾ ∀ l = 1,2,...,(L-1).
/// @param X Input feature matrix X.
/// @param Y Input label matrix Y.

double NeuralNetwork::cost(const vector<mat> Theta, const mat X, const mat Y)
{
    double cost;
    vec error = zeros<vec>(1);
    double regu = 0;

    unsigned int m = X.n_rows;
    vec one = ones<vec>(Y.col(0).n_rows);
    vec h_x;
    vec y_i;
    mat theta;

    //--Error--//
    //-- m K                                                      --//
    //-- ∑ ∑ y⁽i⁾_k log(hΘ(x⁽i⁾)_k) + (1-y⁽i⁾_k) log(1-hΘ(x⁽i⁾)_k)--//
    //-- i k                                                      --//
    for(unsigned int i=0; i<m; i++)
    {
        h_x = h_Theta(X.row(i).t());
        y_i = Y.col(i);

        error = error + ((y_i.t() * log(h_x)) + ((one - y_i).t() * log(one - h_x)));
    }

    //--Regularization--//
    //-- L-1 s_l s_(l+1)           --//
    //--  ∑   ∑     ∑   (Θ⁽l⁾_ij)² --//
    //--  l   i     j              --//
    for(unsigned int l=0; l<Theta.size(); l++)
    {
        theta = Theta[l].cols(1,Theta[l].n_cols-1);
        regu = regu + sum(sum(theta % theta));
        //--Note: '%' is element-wise multiplication in Armadillo--//
    }

    //--       1             λ                  --//
    //-- J(Θ) --- (error) + ---- (Regulization) --//
    //--       m             2m                 --//
    cost = -(sum(error)/m) + ((d_lamda/(2.0*m))*regu);

    return(cost);
}


// void train(const mat, const mat)

/// Trains the neural network through backpropagation and gradient descent, given input matric X and k-class label matric Y.
/// @param X Input feature matrix X.
/// @param Y Input label matrix Y.

void NeuralNetwork::train(const mat X, const mat Y)
{
    if(X.n_rows != Y.n_cols)
    {
        cerr << "ANN-Backpropagation Error: NeuralNetwork class." << endl
             << "void train(const mat, const mat) methon." << endl
             << "Rows of feature matrix X: " << X.n_rows << " should be equal to colums of label matrix Y: " << Y.n_cols
             << endl;

        exit(1);
    }

    vector<mat> partDeriv_cost;
    double c=0;
    double c_prev=0;
    unsigned int it=0;

    fstream costGraph;
    costGraph.open("../Output/cost.dat", ios_base::out);
    costGraph << "#Iteration  #Cost" << endl;

    do
    {
        c_prev = c;
        c = cost(d_Theta, X, Y);
        cout << endl << "Iteration: " << it << "   Cost: " << c << endl;

        //--Cost function graph file update--//
        costGraph << it++ << " " << c << endl;

        //--      ∂          --//
        //--  --------- J(Θ) --//
        //--  ∂ Θ⁽l⁾_ij      --//
        partDeriv_cost = backpropagate(d_Theta, X, Y);

        //--Computer gradient of Θ⁽l⁾_ij ∀ l,i,j, w.r.t J(Θ), and compair it with gradients calculated by backprop to ensure the code is bug-free--//
        //gradientCheck(d_Theta, partDeriv_cost, X, Y);


        //--                       |‾    ∂      ‾|--//
        //-- Θ⁽l⁾_ij := Θ⁽l⁾_ij - 𝛼| --------J(Θ)|--//
        //--                       |_∂Θ⁽l⁾_ij   _|--//
        d_Theta = gradientdescent(d_Theta, partDeriv_cost);

    }while(fabs(c_prev - c) > LEARNING_CURVE_DELTA);

    costGraph.close();
}


// vector<mat> gradientdescent(vector<mat>, const vector<mat>)

/// Implements gradient descent algorithm for updating Θ⁽l⁾_ij ∀ l,i,j, give partial derivatives D⁽l⁾_ij ∀ l,i,j
/// @param Theta A vector of Theta matrices Θ⁽l⁾ ∀ l = 1,2,...,(L-1)
/// @param D A vector of partial derivatives matrices D⁽l⁾ ∀ l = 1,2,...,(L-1)

vector<mat> NeuralNetwork::gradientdescent(vector<mat> Theta, const vector<mat> D)
{
    for(unsigned int l=0; l<Theta.size(); l++)
    {
        //--Θ⁽l⁾_ij := Θ⁽l⁾_ij - 𝛼(∂J(Θ)/∂Θ⁽l⁾_ij)--//
        Theta[l] = Theta[l] - (d_alpha * D[l]);
    }

    return Theta;
}


// vector<mat> backpropagate(vector<mat>, const mat, const mat)

/// Implements backpropagate algorithm for calculating partial derivatives D⁽l⁾_ij ∀ l,i,j
/// Returns a vector of partial derivative matrices D⁽l⁾ ∀ l = 1,2,...,(L-1)
/// @param Theta A vector of Theta matrices Θ⁽l⁾ ∀ l = 1,2,...,(L-1)
/// @param X Input feature matrix X.
/// @param Y Input label matrix Y.

vector<mat> NeuralNetwork::backpropagate(const vector<mat> Theta, const mat X, const mat Y)
{
    //--D⁽l⁾, where l = 1,2,3,...,(L-1)--//
    vector<mat> D = Theta;

    //--Δ⁽l⁾, where l = 1,2,3,...,(L-1)--//
    vector<mat> Delta;

    //--δ⁽l⁾, where l = 2,3,...,(L-1),L--//
    vector<vec> delta(networkArchitecture().size());

    vec a_L;
    vec y_i;
    vec a_l;

    unsigned int m = X.n_rows;
    unsigned int L = networkArchitecture().size()-1;

    //--Set D⁽l⁾_ij = 0, ∀ l,i,j--//
    //--Set Δ⁽l⁾_ij = 0, ∀ l,i,j--//
    for(unsigned int l=0; l<Theta.size(); l++)
    {
        //--D⁽l⁾--//
        D[l].zeros();

        //--Δ⁽l⁾--//
        mat Del = Theta[l];
        Del.zeros();
        Delta.push_back(Del);
    }

    for(unsigned int i=0; i<m; i++)
    {
        //--Forward Propagation--//
        a_L = h_Theta(X.row(i).t());
        y_i = Y.col(i);

        //--δ⁽L⁾ = a⁽L⁾ - y⁽i⁾--//
        delta[L] = a_L - y_i;

        //--Compute δ⁽L-1⁾, δ⁽L-2⁾,...,δ⁽2⁾--//
        for(unsigned int l=L-1; l>0; l--)
        {
            delta[l] = error(Theta[l], delta[l+1], l);
        }

        for(unsigned int l=0; l<L; l++)
        {
            a_l = Layer(l);

            //--Δ⁽l⁾ := Δ⁽l⁾ + (δ⁽l+1⁾ * a⁽l⁾')--//
            Delta[l] = Delta[l] + (delta[l+1] * a_l.t());
        }
    }

    vector<mat> Theta_clone = Theta;
    for(unsigned int l=0; l<L; l++)
    {
        //--Θ⁽l⁾_ij := 0 ∀ j=0--//
        Theta_clone[l].col(0).zeros();

        //--D⁽l⁾ := (¹/m) Δ⁽l⁾ + λΘ⁽l⁾--//
        D[l] = ((1.0/m) * Delta[l]) + (d_lamda * Theta_clone[l]);
    }

    return D;
}


// vec error(const mat, const vec, const unsigned int)

/// Computes error vector δ⁽l⁾, where l=2,3,...,L, given Theta matrix Θ⁽l⁾ and input vector v = δ⁽l+1⁾, if l < L, or v = hΘ(x) if l = L
/// Returns a vector of error terms for layer l.
/// @param Theta matrix Θ⁽l⁾.
/// @param input Input vector v = δ⁽l+1⁾, if l < L, or v = hΘ(x) if l = L.
/// @param l Layer of the network.

vec NeuralNetwork::error(const mat Theta, const vec input, const unsigned int l)
{
    if(l >= networkArchitecture().size())
    {
        cerr << "ANN-Backpropagation Error: NeuralNetwork class." << endl
             << "vec error(const mat, const vec, const unsigned int) methon." << endl
             << "Layer l: " << l << " should be < network architecture size: " << networkArchitecture().size()
             << endl;

        exit(1);
    }

    vec delta_l;
    vec a_l = Layer(l);

    //--δ⁽l⁾ where l=L--//
    if(l == networkArchitecture().size() - 1)
    {
        delta_l = a_l - input;
        return delta_l;
    }
    //--δ⁽l⁾ where l≠L--//
    else
    {
        vec one = ones<vec>(Layer(l).n_rows);

        //--δ⁽l⁾ = (Θ⁽l⁾)'δ⁽l+1⁾ .* g'(z⁽l⁾)--//
        //--Here -> g'(z⁽l⁾) = a⁽l⁾ .* (1-a⁽l⁾) --//
        delta_l = (Theta.t() * input) % (a_l % (one - a_l));

        //--Ommiting δ⁽l⁾_0, which is the error term of the bias node--//
        return delta_l.rows(1,delta_l.n_rows-1);
    }
}


// void gradientCheck(const vector<mat>, const vector<mat>, const mat, const mat)

/// Implements gradient checking by calculating different between partial derivatives calculated numerically and by through backpropagation.
/// @param Theta A vector of Theta matrices Θ⁽l⁾ ∀ l = 1,2,...,(L-1)
/// @param D_backprop A vector of partial derivatives matrices D⁽l⁾ ∀ l = 1,2,...,(L-1), calculated using backpropagation.
/// @param X Input feature matrix X.
/// @param Y Input label matrix Y.

void NeuralNetwork::gradientCheck(const vector<mat> Theta, const vector<mat> D_backprop, const mat X, const mat Y)
{
    vector<mat> D_num;

    D_num = numericalGradient(Theta, X, Y);

    cout << endl;
    for(unsigned int l=0; l<Theta.size(); l++)
    {
        mat diff = D_backprop[l] - D_num[l];
        cout << "Max derivative difference in Theta_" << l+1 << ": " << max(max(diff[l])) << "  ";
    }
    cout << endl;
}


// vector<mat> numericalGradient(const vector<mat>, const mat, const mat)

/// Calculates partial derivatives D⁽l⁾_ij ∀ l,i,j, numerically.
/// Returns a vector of numerically calculated partial derivative matrices D⁽l⁾ ∀ l = 1,2,...,(L-1)
/// @param Theta A vector of Theta matrices Θ⁽l⁾ ∀ l = 1,2,...,(L-1)
/// @param X Input feature matrix X.
/// @param Y Input label matrix Y.

vector<mat> NeuralNetwork::numericalGradient(const vector<mat> Theta, const mat X, const mat Y)
{
    //--D⁽l⁾, where l = 1,2,3,...,(L-1)--//
    vector<mat> D = Theta;

    unsigned int L = networkArchitecture().size()-1;
    double e = 0.001;

    for(unsigned int l=0; l<L; l++)
    {
        for(unsigned int i=0; i<D[l].n_rows; i++)
        {
            for(unsigned int j=0; j<D[l].n_cols; j++)
            {
                vector<mat> Theta_minus_e = Theta;
                vector<mat> Theta_plus_e = Theta;

                Theta_minus_e[l](i,j) = Theta[l](i,j) - e;
                Theta_plus_e[l](i,j) = Theta[l](i,j) + e;

                D[l](i,j) = (cost(Theta_minus_e, X, Y) - cost(Theta_plus_e, X, Y)) / (2.0*e);
            }
        }
    }

    return D;
}
