/********************************************************************************************/
/*                                                                                          */
/*   MLP-Backpropagation: A C++ implementation of Multi-Layer Perceptron (MLP)              */ 
/*                        and backpropagation algorithm for classification                  */
/*                                                                                          */
/*   D A T A   S E T   C L A S S                                                            */
/*                                                                                          */
/*   Avinash Ranganath                                                                      */
/*   Robotics Lab, Department of Systems Engineering and Automation                         */
/*   University Carlos III of Mardid(UC3M)                                                  */
/*   Madrid, Spain                                                                          */
/*   E-mail: nash911@gmail.com                                                              */
/*   https://sites.google.com/site/anashranga/                                              */
/*                                                                                          */
/********************************************************************************************/

#include "data.h"

Data::Data(const char* fileName)
{
    if(!fileName)
    {
        //Error
    }

    unsigned int attSize = attributeSize(fileName);
    unsigned int instSize = instanceSize(fileName);
    unsigned int k = classSize(fileName, instSize, attSize);

    YClass(fileName, instSize, attSize).print();

    d_X.set_size(instSize, attSize);
    d_X.zeros();
    extractX(fileName, instSize, attSize);

    d_Y.set_size(k, instSize);
    d_Y.zeros();
    createYMat(fileName, instSize, attSize);

    cout << endl << "Rows: " << d_X.n_rows << "  Cols: " << d_X.n_cols <<  "   K: " << k << endl;
    //d_X.row(4700).print();

}


// unsigned int get_num_signals(const char* const) method

/// This method extracts and returns the number of signals on the data file.
/// @param signalsFileName Path and name of the file containing signals data.

unsigned int Data::attributeSize(const char* const fileName) const
{
    double numAttributes = 0;

    std::fstream inputFile;
    inputFile.open(fileName, std::ios::in);

    if(!inputFile.is_open())
    {
        std::cerr << "SignalAnalyzer Error: SignalAnalyzerList class." << std::endl
                  << "get_num_signals(const char* const) method" << std::endl
                  << "Cannot open Parameter file: "<< fileName  << std::endl;

        exit(1);
    }

    std::string line;
    string attribute;
    std::size_t found;

    //--Omitting lines containing '#'--//
    do
    {
        getline(inputFile, line);
        found = line.find("#");
    }while(found != std::string::npos);

    std::stringstream ssLine(line);
    ssLine >> attribute;

    //--Extracting the number of signals on file--//
    while(ssLine >> attribute)
    {
        //cout << attribute << " ";
        numAttributes++;
    }


    std::cout << std::endl << "Number of attributes of the data set: " << numAttributes << std::endl << std::endl;
    inputFile.close();

    return numAttributes;
}



// unsigned int get_num_signals(const char* const) method

/// This method extracts and returns the number of signals on the data file.
/// @param signalsFileName Path and name of the file containing signals data.

unsigned int Data::instanceSize(const char* const fileName) const
{
    double numInstances = 0;

    std::fstream inputFile;
    inputFile.open(fileName, std::ios::in);

    if(!inputFile.is_open())
    {
        std::cerr << "SignalAnalyzer Error: SignalAnalyzerList class." << std::endl
                  << "get_num_signals(const char* const) method" << std::endl
                  << "Cannot open Parameter file: "<< fileName  << std::endl;

        exit(1);
    }

    std::string line;
    std::size_t found;

    //--Omitting lines containing '#'--//
    do
    {
        getline(inputFile, line);
        found = line.find("#");
    }while(found != std::string::npos);

    //--Extracting the number of signals on file--//
    while(!inputFile.eof())
    {
        getline(inputFile, line);
        numInstances++;
    }


    std::cout << std::endl << "Number of instances on the file: " << numInstances << std::endl << std::endl;
    inputFile.close();

    return numInstances;
}

// unsigned int get_num_signals(const char* const) method

/// This method extracts and returns the number of signals on the data file.
/// @param signalsFileName Path and name of the file containing signals data.

unsigned int Data::classSize(const char* const fileName, const unsigned int M, const unsigned int N) const
{
    return YClass(fileName, M, N).n_rows;
}


// unsigned int get_num_signals(const char* const) method

/// This method extracts and returns the number of signals on the data file.
/// @param signalsFileName Path and name of the file containing signals data.

vec Data::YClass(const char* const fileName, const unsigned int M, const unsigned int N) const
{
    return unique(extractY(fileName, M, N));
}


void Data::extractX(const char* const fileName, const unsigned int M, const unsigned int N)
{
    std::fstream inputFile;
    inputFile.open(fileName, std::ios::in);

    if(!inputFile.is_open())
    {
        std::cerr << "SignalAnalyzer Error: SignalAnalyzerList class." << std::endl
                  << "get_num_signals(const char* const) method" << std::endl
                  << "Cannot open Parameter file: "<< fileName  << std::endl;

        exit(1);
    }


    std::string line;
    double X_i;
    std::size_t found;

    //--Omitting lines containing '#'--//
    do
    {
        getline(inputFile, line);
        found = line.find("#");
    }while(found != std::string::npos);

    //--Extracting the number of signals on file--//
    for(unsigned int m=0; m<M; m++)
    {
        //cout << m+1;
        std::stringstream ssLine(line);
        for(unsigned int n=0; n<N; n++)
        {
            ssLine >> X_i;
            d_X(m,n) = X_i;
            //cout << " " << X_i;
        }
        //cout << endl;
        getline(inputFile, line);
    }

}


vec Data::extractY(const char* const fileName, const unsigned int M, const unsigned int N) const
{
    std::fstream inputFile;
    inputFile.open(fileName, std::ios::in);

    if(!inputFile.is_open())
    {
        std::cerr << "SignalAnalyzer Error: SignalAnalyzerList class." << std::endl
                  << "get_num_signals(const char* const) method" << std::endl
                  << "Cannot open Parameter file: "<< fileName  << std::endl;

        exit(1);
    }

    vec Y = zeros<vec>(M);

    std::string line;
    double X_i;
    double y;
    std::size_t found;

    //--Omitting lines containing '#'--//
    do
    {
        getline(inputFile, line);
        found = line.find("#");
    }while(found != std::string::npos);

    //--Extracting the number of signals on file--//
    for(unsigned int m=0; m<M; m++)
    {
        std::stringstream ssLine(line);
        for(unsigned int n=0; n<N; n++)
        {
            ssLine >> X_i;
        }
        ssLine >> y;
        Y(m) = y;

        getline(inputFile, line);
    }

    return Y;
}


void Data::createYMat(const char* const fileName, const unsigned int M, const unsigned int N)
{
    vec Y = extractY(fileName, M, N);
    vec yClass = unique(Y);
    //unsigned int indx = 0;

    for(unsigned int m=0; m<M; m++)
    {
        uvec indx = find(yClass == Y[m], 1);
        d_Y(indx[0], m) = 1.0;
    }

    //d_Y.cols(0,7).print();
}


unsigned int Data::M() const
{
    return d_X.n_rows;
}


unsigned int Data::N() const
{
    return d_X.n_cols;
}


unsigned int Data::K() const
{
    return d_Y.n_rows;
}


mat Data::X(void) const
{
    return d_X;
}


mat Data::Y(void) const
{
    return d_Y;
}
