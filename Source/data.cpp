/********************************************************************************************/
/*                                                                                          */
/*   ANN-Backpropagation: A C++ implementation of Artificial Neural Network (ANN)           */
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


// CONSTRUCTOR

/// Creates a Data Set object.
/// Extracts training data containing features and targets from the file whose path and name is passed as a parameter.
/// Shuffels the data set and divides it into training and test sets.
/// @param fileName Path and name of the file containing the training data.
/// @param trainPercent Training split of the data set > 0%.
/// @param testPercent Test split of the data set ≥ 0%.

Data::Data(const char* fileName, const double trainPercent, const double testPercent)
{
    if(!fileName)
    {
        cerr << "ANN-Backpropagation Error: Data class." << endl
             << "DataSet(const char*, const double, const double) constructor." << endl
             << "Cannot open data file: " << fileName
             << endl;

        exit(1);
    }

    //--Extract no. of instances(m), attributes(n) and classes(k) in the data set on file--//
    unsigned int attSize = attributeSize(fileName);
    unsigned int instSize = instanceSize(fileName);
    unsigned int k = classSize(fileName, instSize, attSize);

    //--Print the k classes--//
    YClass(fileName, instSize, attSize).print();

    //--Extract feature set from data file--//
    d_X.set_size(instSize, attSize);
    d_X.zeros();
    extractX(fileName, instSize, attSize);

    //--Extract targets from data file and create matrix Y, where y⁽i⁾ ∈ R^k (Eg: y⁽i⁾ = [... 0 1 0 ...]')--//
    d_Y.set_size(k, instSize);
    d_Y.zeros();
    createYMat(fileName, instSize, attSize);

    cout << endl << "Rows: " << d_X.n_rows << "  Cols: " << d_X.n_cols <<  "   K: " << k << endl;
}


// unsigned int attributeSize(const char* const) const method

/// Extracts and returns the number of features (n) on the data file.
/// @param fileName Path and name of the file containing the training data.

unsigned int Data::attributeSize(const char* const fileName) const
{
    double numAttributes = 0;

    fstream inputFile;
    inputFile.open(fileName, ios::in);

    if(!inputFile.is_open())
    {
        cerr << "ANN-Backpropagation Error: Data class." << endl
             << "unsigned int attributeSize(const char* const) const method." << endl
             << "Cannot open Parameter file: "<< fileName  << endl;

        exit(1);
    }

    string line;
    string attribute;
    size_t found;

    //--Omitting lines containing '#'--//
    do
    {
        getline(inputFile, line);
        found = line.find("#");
    }while(found != string::npos);

    stringstream ssLine(line);
    ssLine >> attribute;

    //--Extracting the number of signals on file--//
    while(ssLine >> attribute)
    {
        numAttributes++;
    }

    cout << endl << "Number of attributes of the data set: " << numAttributes << endl << endl;
    inputFile.close();

    return numAttributes;
}


// unsigned int instanceSize(const char* const) const method

/// Extracts and returns the number of instances (m) on the data file.
/// @param fileName Path and name of the file containing the training data.

unsigned int Data::instanceSize(const char* const fileName) const
{
    double numInstances = 0;

    fstream inputFile;
    inputFile.open(fileName, ios::in);

    if(!inputFile.is_open())
    {
        cerr << "ANN-Backpropagation Error: Data class." << endl
             << "unsigned int instanceSize(const char* const) const method." << endl
             << "Cannot open Parameter file: "<< fileName  << endl;

        exit(1);
    }

    string line;
    size_t found;

    //--Omitting lines containing '#'--//
    do
    {
        getline(inputFile, line);
        found = line.find("#");
    }while(found != string::npos);

    //--Extracting the number of signals on file--//
    while(!inputFile.eof())
    {
        getline(inputFile, line);
        numInstances++;
    }


    cout << endl << "Number of instances on the file: " << numInstances << endl << endl;
    inputFile.close();

    return numInstances;
}


// unsigned int classSize(const char* const, const unsigned int, const unsigned int) const method

/// Returns the size of vector of unique labels (k) on the data file.
/// @param fileName Path and name of the file containing training data.
/// @param M Number of instances on file.
/// @param N Number of attributes on file.

unsigned int Data::classSize(const char* const fileName, const unsigned int M, const unsigned int N) const
{
    return YClass(fileName, M, N).n_rows;
}


// unsigned int YClass(const char* const, const unsigned int, const unsigned int) const method

/// Returns a vector of unique labels (Y_i) on the data file.
/// @param fileName Path and name of the file containing training data.
/// @param M Number of instances on file.
/// @param N Number of attributes on file.

vec Data::YClass(const char* const fileName, const unsigned int M, const unsigned int N) const
{
    return unique(extractY(fileName, M, N));
}


// void extractX(const char* const, const unsigned int, const unsigned int) method

/// Extracts attributes of the data set from the file.
/// @param fileName Path and name of the file containing training data.
/// @param M Number of instances on file.
/// @param N Number of attributes on file.

void Data::extractX(const char* const fileName, const unsigned int M, const unsigned int N)
{
    fstream inputFile;
    inputFile.open(fileName, ios::in);

    if(!inputFile.is_open())
    {
        cerr << "ANN-Backpropagation Error: Data class." << endl
             << "void extractX(const char* const, const unsigned int, const unsigned int) method." << endl
             << "Cannot open Parameter file: "<< fileName  << endl;

        exit(1);
    }


    string line;
    double X_i;
    size_t found;

    //--Omitting lines containing '#'--//
    do
    {
        getline(inputFile, line);
        found = line.find("#");
    }while(found != string::npos);

    //--Extracting the number of signals on file--//
    for(unsigned int m=0; m<M; m++)
    {
        //cout << m+1;
        stringstream ssLine(line);
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


// void extractY(const char* const, const unsigned int, const unsigned int) method

/// Extracts targets of the data set from the file.
/// @param fileName Path and name of the file containing training data.
/// @param M Number of instances on file.
/// @param N Number of attributes on file.

vec Data::extractY(const char* const fileName, const unsigned int M, const unsigned int N) const
{
    fstream inputFile;
    inputFile.open(fileName, ios::in);

    if(!inputFile.is_open())
    {
        cerr << "ANN-Backpropagation Error: Data class." << endl
             << "vec extractY(const char* const, const unsigned int, const unsigned int) const method." << endl
             << "Cannot open Parameter file: "<< fileName  << endl;

        exit(1);
    }

    vec Y = zeros<vec>(M);

    string line;
    double X_i;
    double y;
    size_t found;

    //--Omitting lines containing '#'--//
    do
    {
        getline(inputFile, line);
        found = line.find("#");
    }while(found != string::npos);

    //--Extracting the number of signals on file--//
    for(unsigned int m=0; m<M; m++)
    {
        stringstream ssLine(line);
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


// void createYMat(const char* const, const unsigned int, const unsigned int) method

/// Based on the target vector y ∈ R^m, extracts unique target values.
/// Creates matrix Y ∈ R^(kxm), where y⁽i⁾ ∈ R^k, and of type [... 0 1 0 ...]'
/// @param fileName Path and name of the file containing training data.
/// @param M Number of instances on file.
/// @param N Number of attributes on file.

void Data::createYMat(const char* const fileName, const unsigned int M, const unsigned int N)
{
    vec Y = extractY(fileName, M, N);
    vec yClass = unique(Y);

    for(unsigned int m=0; m<M; m++)
    {
        uvec indx = find(yClass == Y[m], 1);
        d_Y(indx[0], m) = 1.0;
    }
}


// unsigned int M() const method

/// Returns the number of rows in the data set.

unsigned int Data::M() const
{
    return d_X.n_rows;
}


// unsigned int N() const method

/// Returns the number of colums in the data set.

unsigned int Data::N() const
{
    return d_X.n_cols;
}


// unsigned int K() const method

/// Returns the number of classes in the data set.

unsigned int Data::K() const
{
    return d_Y.n_rows;
}

// mat X() const method

/// Returns matrix X of the data set.

mat Data::X(void) const
{
    return d_X;
}

// mat Y() const method

/// Returns matrix Y of the data set.

mat Data::Y(void) const
{
    return d_Y;
}
