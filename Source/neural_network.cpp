/********************************************************************************************/
/*                                                                                          */
/*   MLP-Backpropagation: A C++ implementation of Multi-Layer Perceptron (MLP)              */ 
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

NeuralNetwork::NeuralNetwork(const unsigned int inputNeurons, const vector<unsigned int> hiddenLayers, const unsigned int outputNeurons)
{
    d_alpha = 0.3;
    d_lamda = 0.0;

    d_inputneurons.set_size(inputNeurons);
    d_outputneurons.set_size(outputNeurons);

    for(unsigned int hl = 0; hl < hiddenLayers.size(); hl++)
    {
        vec hidLayer(hiddenLayers[hl]);
        d_hiddenneurons.push_back(hidLayer);
    }

    unsigned int rows = 0;
    unsigned int cols = 0;
    //unsigned int layers = hiddenLayers.size() + 2;
    unsigned int layers = networkArchitecture().size();

    for(unsigned int l = 0; l < (layers - 1); l++)
    {
        rows = networkArchitecture()[l+1];
        cols = networkArchitecture()[l]+1;

        cout << endl << "Layer " << l+1 <<  ":    Rows: " << rows << "    Cols: " << cols << endl;

        mat theta(rows, cols, fill::randu);
        d_Theta.push_back(theta);
    }
}


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


vec NeuralNetwork::Layer(const unsigned int l) const
{
    if(l >= networkArchitecture().size())
    {
        //Error
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


void NeuralNetwork::printTheta(void) const
{
    for(unsigned int l=0; l<d_Theta.size(); l++)
    {
        cout << endl << "Theta_" << l+1 << ":" << endl;
        d_Theta[l].print();
    }
}


vec NeuralNetwork::activation(const vec inputVec, const unsigned int l)
{
    if(l >= networkArchitecture().size())
    {
        //Error
    }

    vec a = inputVec;

    if(l > 0)
    {
        a = sigmoid(d_Theta[l-1] * a);
    }

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


    if(l < networkArchitecture().size() - 1)
    {
        //--Adding element a_0 ∀ l≠L--//
        vec a_0 = ones(1);
        a.insert_rows(0, a_0);
    }

    return a;
}


vec NeuralNetwork::sigmoid(const vec z)
{
    vec one = ones<vec>(z.n_rows);

    return(one/(one + exp(-z)));

}


vec NeuralNetwork::h(const vec x)
{
    vec a = x;
    unsigned int layers = networkArchitecture().size();

    for(unsigned int l=0; l<layers; l++)
    {
        a = activation(a, l);
    }

    return(a);
}


double NeuralNetwork::cost(const mat x, const mat y)
{
    double cost;
    vec error = zeros<vec>(1);
    double regu = 0;

    unsigned int m = x.n_rows;
    vec one = ones<vec>(y.col(0).n_rows);
    vec h_x;
    vec y_i;
    mat Theta;

    for(unsigned int i=0; i<m; i++)
    {
        h_x = h(x.row(i).t());
        y_i = y.col(i);

        error = error + ((y_i.t() * log(h_x)) + ((one - y_i).t() * log(one - h_x)));
    }

    for(unsigned int l=0; l<d_Theta.size(); l++)
    {
        Theta = d_Theta[l].cols(1,d_Theta[l].n_cols-1);
        regu = regu + sum(sum(Theta % Theta));
    }

    cost = -(sum(error)/m) + ((d_lamda/(2.0*m))*regu);

    return(cost);
}


void NeuralNetwork::gradientdescent(const mat x, const mat y)
{
    if(x.n_rows != y.n_cols)
    {
        //Error
    }

    vector<mat> partDeriv_cost;
    double c=0;
    double c_prev=0;
    unsigned int it=0;

    fstream costGraph;
    costGraph.open("../Output/cost.dat", ios_base::out);
    costGraph << "#Iteration  #Cost" << endl;

    //for(unsigned int i=0; i<50; i++)
    do
    {
        c_prev = c;
        c = cost(x, y);
        //cout << endl << "Iteration: " << it << "   Cost: " << c << endl;
        costGraph << it++ << " " << c << endl;

        //--      ∂          --//
        //--  --------- J(Θ) --//
        //--  ∂ Θ⁽l⁾_ij      --//
        partDeriv_cost = backpropagate(x, y);
        //cout << endl << "Finished BackProp() " << endl;

        for(unsigned int l=0; l<d_Theta.size(); l++)
        {
            //--Θ⁽l⁾_ij := Θ⁽l⁾_ij - 𝛼(∂J(Θ)/∂Θ⁽l⁾_ij)--//
            d_Theta[l] = d_Theta[l] - (d_alpha * partDeriv_cost[l]);
        }
    }while(fabs(c_prev - c) > DELTA);

    costGraph.close();
}


vector<mat> NeuralNetwork::backpropagate(const mat x, const mat y)
{
    vector<mat> D = d_Theta;
    vector<mat> Delta;
    vector<vec> delta(networkArchitecture().size());

    vec h_x;
    vec y_i;
    vec a;

    unsigned int m = x.n_rows;
    unsigned int L = networkArchitecture().size()-1;

    for(unsigned int l=0; l<d_Theta.size(); l++)
    {
        mat Del = d_Theta[l];
        Del.zeros();
        Delta.push_back(Del);

        D[l].zeros();

    }

    for(unsigned int i=0; i<m; i++)
    {
        //--Forward Propagation--//
        h_x = h(x.row(i).t());
        y_i = y.col(i);

        //cout << endl << "Finished ForwardProp() " << endl;
        delta[L] = h_x - y_i;
        for(unsigned int l=L-1; l>0; l--)
        {
            delta[l] = error(delta[l+1], l);
        }

        for(unsigned int l=0; l<L; l++)
        {
            a = Layer(l);
            Delta[l] = Delta[l] + (delta[l+1] * Layer(l).t());
        }
    }

    vector<mat> Theta = d_Theta;
    for(unsigned int l=0; l<L; l++)
    {
        //--∀ Θ⁽l⁾_ij if j=0--//
        Theta[l].col(0).zeros();

        //--D⁽l⁾ := (¹/m) Δ⁽l⁾ + λΘ⁽l⁾--//
        D[l] = ((1.0/m) * Delta[l]) + (d_lamda * Theta[l]);
    }

    return D;
}


vec NeuralNetwork::error(const vec input, const unsigned int l)
{
    if(l >= networkArchitecture().size())
    {
        //Error
    }

    vec delta;
    vec a = Layer(l);

    //--δ⁽l⁾ where l=L--//
    if(l == networkArchitecture().size() - 1)
    {
        delta = a - input;
        return delta;
    }
    //--δ⁽l⁾ where l≠L--//
    else
    {
        vec one = ones<vec>(Layer(l).n_rows);
        delta = (d_Theta[l].t() * input) % (a % (one - a));
        return delta.rows(1,delta.n_rows-1);
    }
    //cout << endl << "Finished error() " << endl;
}
