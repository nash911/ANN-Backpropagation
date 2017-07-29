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

Data::Data(const char* fileName, const double& trainPercent, const double& testPercent)
{
    if(!fileName)
    {
        cerr << "ANN-Backpropagation Error: Data class." << endl
             << "DataSet(const char*, const double, const double) constructor." << endl
             << "Cannot open data file: " << fileName
             << endl;

        exit(1);
    }

    if(trainPercent <= 0.0 || testPercent < 0.0)
    {
      cerr << "ANN-Backpropagation Error: Data class." << endl
           << "DataSet(const char*, const double, const double) constructor." << endl
             << "Training set = " << trainPercent << "% has to be > 0% and Test set = "<< testPercent << "% has to be >= 0%."
             << endl;

        exit(1);
    }

    if(trainPercent + testPercent != 100.0)
    {
      cerr << "ANN-Backpropagation Error: Data class." << endl
           << "DataSet(const char*, const double, const double) constructor." << endl
             << "Training set = " << trainPercent << "% + Test set = "<< testPercent << "% has to be equal to 100%!"
             << endl;

        exit(1);
    }

    //--Extract no. of instances(m), attributes(n) and classes(k) in the data set on file--//
    unsigned int attSize = attributeSize(fileName);
    unsigned int instSize = instanceSize(fileName);
    unsigned int k = classSize(fileName, instSize, attSize);

    //--Print the k classes--//
    cout << endl << "k-class labels:" << endl;
    YClass(fileName, instSize, attSize).print();

    //--Extract feature set from data file--//
    d_X.set_size(instSize, attSize);
    d_X.zeros();
    extractX(fileName, instSize, attSize);

    //--Extract targets from data file and create matrix Y, where y⁽i⁾ ∈ R^k (Eg: y⁽i⁾ = [... 0 1 0 ...]')--//
    d_Y.set_size(k, instSize);
    d_Y.zeros();
    createYMat(fileName, instSize, attSize);

    //--Shuffle the data and segment into training and test sets--//
    segmentDataSet(trainPercent, testPercent);

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


// unsigned int classSize(const char* const, const unsigned int&, const unsigned int&) const method

/// Returns the size of vector of unique labels (k) on the data file.
/// @param fileName Path and name of the file containing training data.
/// @param M Number of instances on file.
/// @param N Number of attributes on file.

unsigned int Data::classSize(const char* const fileName, const unsigned int& M, const unsigned int& N) const
{
    return YClass(fileName, M, N).n_rows;
}


// unsigned int YClass(const char* const, const unsigned int&, const unsigned int&) const method

/// Returns a vector of unique labels (Y_i) on the data file.
/// @param fileName Path and name of the file containing training data.
/// @param M Number of instances on file.
/// @param N Number of attributes on file.

vec Data::YClass(const char* const fileName, const unsigned int& M, const unsigned int& N) const
{
    return unique(extractY(fileName, M, N));
}


// void extractX(const char* const, const unsigned int&, const unsigned int&) method

/// Extracts attributes of the data set from the file.
/// @param fileName Path and name of the file containing training data.
/// @param M Number of instances on file.
/// @param N Number of attributes on file.

void Data::extractX(const char* const fileName, const unsigned int& M, const unsigned int& N)
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


// void extractY(const char* const, const unsigned int&, const unsigned int&) method

/// Extracts targets of the data set from the file.
/// @param fileName Path and name of the file containing training data.
/// @param M Number of instances on file.
/// @param N Number of attributes on file.

vec Data::extractY(const char* const fileName, const unsigned int& M, const unsigned int& N) const
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


// void createYMat(const char* const, const unsigned int&, const unsigned int&) method

/// Based on the target vector y ∈ R^m, extracts unique target values.
/// Creates matrix Y ∈ R^(kxm), where y⁽i⁾ ∈ R^k, and of type [... 0 1 0 ...]'
/// @param fileName Path and name of the file containing training data.
/// @param M Number of instances on file.
/// @param N Number of attributes on file.

void Data::createYMat(const char* const fileName, const unsigned int& M, const unsigned int& N)
{
    vec Y = extractY(fileName, M, N);
    vec yClass = unique(Y);

    for(unsigned int m=0; m<M; m++)
    {
        uvec indx = find(yClass == Y[m], 1);
        d_Y(indx[0], m) = 1.0;
    }
}


// void segmentDataSet(const double&, const double&) const method

/// Shuffels the data set and divides it into training and test sets.
/// @param trainPercent Training split of the data set > 0%.
/// @param testPercent Test split of the data set ≥ 0%.

void Data::segmentDataSet(const double& trainPercent, const double& testPercent)
{
    if(!(trainPercent >= 0.0 && trainPercent <= 100.0) || !(testPercent >= 0.0 && testPercent <= 100.0))
    {
        cerr << "ANN-Backpropagation Error: Data class." << endl
             << "void segmentDataSet(const double, const double) method" << endl
             << "Training size(%): " << trainPercent << " and Test size(%): " << testPercent << " should both be in the range [0,100]."
             << endl;

        exit(1);
    }

    if(trainPercent + testPercent != 100.0)
    {
        cerr << "ANN-Backpropagation Error: Data class." << endl
             << "void segmentDataSet(const double, const double) method" << endl
             << "Training set: " << trainPercent << " + Test set: " << testPercent << " != 100%"
             << endl;

        exit(1);
    }

    unsigned int m = d_X.n_rows;
    unsigned int n = d_X.n_cols;
    unsigned int k = d_Y.n_rows;

    //--Combine matrix X and vector y by inserting vector y as the last column of matrix X--//
    mat Xy = d_X;
    Xy.insert_cols(n, d_Y.t());

    //--Shuffle the whole data set--//
    Xy = shuffle(Xy);

    //--Calculate training set and test set size--//
    unsigned int trainSize = m * (trainPercent/100.0);
    unsigned int testSize = m - trainSize;

    //--Segregate data into training and test sets--//
    if(trainSize)
    {
        d_X = Xy;
        d_X.shed_rows(trainSize, m-1);

        d_Y = (d_X.cols(n, n+k-1)).t();
        d_X.shed_cols(n, n+k-1);
    }

    if(testSize)
    {
        d_X_test = Xy;
        if(trainSize)
        {
            d_X_test.shed_rows(0, trainSize-1);
        }

        d_Y_test = (d_X_test.cols(n, n+k-1)).t();
        d_X_test.shed_cols(n, n+k-1);
    }

    if(d_X.n_rows + d_X_test.n_rows != m)
    {
        cerr << "ANN-Backpropagation Error: Data class." << endl
             << "void segmentDataSet(const double, const double) method" << endl
             << "Training set: " << d_X.n_rows << " + Test set: " << d_X.n_rows << " != Data size: " << m
             << endl;

        exit(1);
    }

    cout << endl << "Training set size: " << d_X.n_rows
         << endl << "Test set size: " << d_X_test.n_rows << endl;
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


// mat X_test() const method

/// Returns matrix X of the test data set.

mat Data::X_test(void) const
{
    return d_X_test;
}


// mat Y_test() const method

/// Returns matrix Y of the test data set.

mat Data::Y_test(void) const
{
    return d_Y_test;
}
