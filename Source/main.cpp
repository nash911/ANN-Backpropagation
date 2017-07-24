#include "data.h"
#include "neural_network.h"

#define TRAINING_SIZE 70.0
#define TEST_SIZE 30.0

int main(int argc, char* argv[])
{
    //--Initializing random seed--//
    srand (time(NULL));

    unsigned int hiddenLayers = 0;
    unsigned int hiddenLayer_size = 0;

    char* dataFileName;
    fstream dataFile;

    if(argc == 1)
    {
        hiddenLayers = 2;
    }
    else if(argc >= 3)
    {
        hiddenLayers = atoi(argv[1]);
        hiddenLayer_size = atoi(argv[2]);
    }

    if(argc == 4)
    {
        dataFileName = argv[3];
    }
    else
    {
        dataFileName = "../Data/winequality.csv";
    }

    dataFile.open(dataFileName, ios_base::in);
    if(!dataFile.is_open())
    {
        //Error
    }
    dataFile.close();

    Data d(dataFileName, TRAINING_SIZE, TEST_SIZE);

    vector<unsigned int> hidLayer;
    hidLayer.push_back(d.N());
    hidLayer.push_back(d.N());

    NeuralNetwork n(d.N(), hidLayer, d.K());

    //n.gradientdescent(d.X(), d.Y());
    n.train(d.X(), d.Y());

    cout <<endl << "Exiting Successfully" << endl;

    return 0;
}
