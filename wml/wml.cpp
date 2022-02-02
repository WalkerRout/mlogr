// Author: Walker Rout
// Date Created: January 24th, 2022



/* Include Statements */
/* ------------------ */

#include <iostream>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include "wml.h"

/* ------------------ */





// Purpose       : Print out a matrix in order of rows and columns, separate by spaces and newlines respectively
// Parameters    : mat {std::vector<std::vector<double>>} - Matrix to print
// Return values : N/A
void ML::printMat(std::vector< std::vector<double> > mat){
  for(int i = 0; i < mat.size(); i++){
    for(int j = 0; j < mat[i].size(); j++){
      printf("%f ", mat[i][j]);
    }
    printf("\n");
  }
}



// Purpose       : Print out a vector by rows separated by spaces
// Parameters    : vec {std::vector<double>} - Vector to print
// Return values : N/A
void ML::printVec(std::vector<double> vec){
  for(int i = 0; i < vec.size(); i++){
    printf("%f ", vec[i]);
  }
  printf("\n");
}



// Purpose       : Read a CSV file into a matrix (no words or NaN values can be contained in the CSV, all rows must be filled)
// Parameters    : filename {std::string} - The path of the file to read in (ie.. data/example.csv)
// Return values : {std::vector<std::vector<double>>} - A matrix containing all of the data in the CSV file
std::vector< std::vector<double> > ML::readCSV(std::string filename){
  std::vector< std::vector<double> > mat;

  std::ifstream fin(filename);
  std::string linestr;

  while(std::getline(fin, linestr)){
    std::stringstream ssr(linestr);
    std::stringstream ssc;
    std::vector<double> temp;
    std::string data;
    double t;

    
    while(std::getline(ssr, data, ',')){
      ssc << data;
      ssc >> t;
      temp.push_back(t);
      std::stringstream().swap(ssc);
    }

    if (temp.size() > 0) mat.push_back(temp);
  }

  return mat;
}



// Purpose       : Split the binary outcome from an input matrix into a separate vector and append a 1 to the front of all rows in the input matrix (binary outcome must be the last column of the input matrix)
// Parameters    : X {std::vector<std::vector<double>> &} - Input matrix to split and append, y_i {std::vector<double> &} - Vector to move the binary outcome column into 
// Return values : N/A
void ML::splitVariables(std::vector< std::vector<double> > &X, std::vector<double> &y_i){
  for(auto &x_i : X){
    y_i.push_back(x_i[x_i.size() - 1]);
    x_i.pop_back();
    x_i.insert(x_i.begin(), 1); // Easier dot product calculations after appending a 1 to the beginning
  }
}



// Purpose       : Calculate the accuracy of a Logistic Regression model based on true positives, true negatives, false positives, and false negatives
// Parameters    : log {ML::LogisticRegression} - Model object (can be any model), X {std::vector<std::vector<double>>} - Input matrix of values, y_i {std::vector<double>} - Vector of true binary outcomes, b_i {std::vector<double>} - Updated intercept and weights of the model, total {double} - Size of the y_i vector, thresh {double} - Threshold for a 0 or 1 prediction
// Return values : N/A
void ML::accuracy(ML::LogisticRegression logr, std::vector< std::vector<double> > X, std::vector<double> y_i, std::vector<double> b_i, double thresh){
  double total = y_i.size();
  
  double tpos = 0;
  double tneg = 0;
  double fpos = 0;
  double fneg = 0;
  
  for(int i = 0; i < X.size(); i++){
    std::vector<double> x_i = X[i]; 
    double y = y_i[i];
    double y_hat = logr.predict(x_i, b_i);
    double y_pred = 1.0;

    if(y_hat < thresh){
      y_pred = 0.0;
    }

    if(y_pred == 1.0 && y == 1.0){
      tpos += 1.0;
    } else if (y_pred == 0.0 && y == 0.0){
      tneg += 1.0;
    } else if (y_pred == 1.0 && y == 0.0){
      fpos += 1;
    } else if (y_pred == 0.0 && y == 1.0){
      fneg += 1;
    } else {
      printf("Invalid y_pred and y value match!\n");
    }
  }

  printf("\nAccuracy: %.2f percent!\n", (((tpos + tneg) / total) * 100));
}



// Purpose       : Transpose a matrix (columns -> rows, rows -> columns)
// Parameters    : mat {std::vector<std::vector<double>>} - Matrix to transpose
// Return values : {std::vector<std::vector<double>>} - The transposed matrix based on the mat parameter
std::vector< std::vector<double> > ML::transpose(std::vector< std::vector<double> > mat){
  std::vector< std::vector<double> > mat_n(mat[0].size(), std::vector<double>(mat.size(), 0.0));
  
  for(int i = 0; i < mat.size(); i++){
    for (int j = 0; j < mat[i].size(); j++){
      mat_n[j][i] = mat[i][j];
    }
  }

  return mat_n;
}



// Purpose       : Find the mean value of a vector (sum of elements / number of elements)
// Parameters    : x_i {std::vector<double>} - Vector to find the mean of
// Return values : {double} - The mean of the vector
double ML::mean(std::vector<double> x_i){
  double sum = 0.0;

  for(auto x_ij : x_i){
    sum += x_ij;
  }

  return sum / x_i.size();
}



// Purpose       : Find the standard deviation of a vector given by formula sqrt(sum[(ele - mean)**2] / num_ele - 1) where ele is an element in the vector
// Parameters    : x_i {std::vector<double>} - Vector to calculate standard deviation of
// Return values : {double} - Standard deviation of the x_i parameter
double ML::std_dev(std::vector<double> x_i){
  size_t num_xs = x_i.size();
  double mn = mean(x_i);
  double sum_diff = 0.0;

  for(auto x_ij : x_i){
    sum_diff += pow((x_ij - mn), 2.0);
  }

  return sqrt(1.0 / (num_xs - 1.0) * sum_diff);
}



// Purpose       : Standardize a matrix based on the transpose of that matrix. This method is to be used with matrices that have had a 1 appended to their inner vectors
// Parameters    : X {std::vector<std::vector<double>>} - Matrix to standardize
// Return values : {std::vector<std::vector<double>>} - The standardized version of the X parameter
std::vector< std::vector<double> > ML::zscore(std::vector< std::vector<double> > X){
  std::vector< std::vector<double> > XT = transpose(X);
  std::vector<double> mns;
  std::vector<double> stds;

  for(auto x_i : XT){
    mns.push_back(mean(x_i));
    stds.push_back(std_dev(x_i));
  }

  for(int i = 0; i < X.size(); i++){
    for(int j = 0; j < X[0].size(); j++){
      if(stds[j] > 0.0){
        X[i][j] = ((X[i][j] - mns[j]) / stds[j]);
      }
    }
  }

  return X;
}



// Purpose       : Fill a vector with random numbers between 0 and 1
// Parameters    : vec {std::vector<double> &} - A vector with defined size to fill with random numbers
// Return values : N/A
void ML::fill_rand(std::vector<double> &vec){
  srand(time(0));
  std::generate(vec.begin(), vec.end(), rand);
  std::for_each(vec.begin(), vec.end(), [] (double &ele){ele /= RAND_MAX;} );
}



// Class method initializations

// Purpose       : Constructor method for LogisticRegression class instances
// Parameters    : epochs {unsigned int} - Amount of epochs (iterations) of the gradient descent algorithm, learning_rate {double} - Learning rate for the model (typically 0.01)
// Return values : N/A
ML::LogisticRegression::LogisticRegression(unsigned int epochs, double learning_rate){
  this->epochs = epochs;
  this->learning_rate = learning_rate;
}




// Purpose       : Calculate the value of a double passed through a logistic curve (sigmoid function)
// Parameters    : z {double} - The value to be passed through the function
// Return values : {double} - The corresponding value of the z paramater passed through the sigmoid function
double ML::LogisticRegression::sigmoid(double z){
  return 1.0 / (1.0 + exp(-z));
}



// Purpose       : Calculate the dot product between a row of the input matrix (a vector) and the beta vector (intercept and weights)
// Parameters    : x_i {std::vector<double} - Vector of independent variables with a 1 appended to the front, b_i {std::vector<double>} - Vector of weights with the intercept appended to the front
// Return values : {double} - The dot product of the x_i parameter and b_i parameter
double ML::LogisticRegression::dot(std::vector<double> x_i, std::vector<double> b_i){
  double prod = 0.0;

  for(int i = 0; i < x_i.size(); i++){
    double x_ij = x_i[i];
    double b = b_i[i];
    prod += x_ij*b;
  }
  
  return prod;
}



// Purpose       : A prediction of the model based on the independent variables and weights found by passing the dot product of the independent variable vectors and weights into the sigmoid method
// Parameters    : x_i {std::vector<double>} - Vector of independent variables with a 1 appended to the front, b_i {std::vector<double>} - Vector of weights with the intercept appended to the front
// Return values : {double} - The prediction of the model based off of the beta values (between 0 and 1)
double ML::LogisticRegression::predict(std::vector<double> x_i, std::vector<double> b_i){
  return sigmoid(dot(x_i, b_i));
}



// Purpose       : Loss function for the actual vs predicted y-value
// Parameters    : y_i {double} - Actual y-value, y_hat {double} - Predicted y-value
// Return values : {double} - Logarithmic loss of the model based on the actual vs predicted value
double ML::LogisticRegression::logloss(double y_i, double y_hat){
  return -((y_i * log(y_hat)) + ((1.0 - y_i) * log(1.0 - y_hat)));
}



// Purpose       : Calculates the total mean error of the function for every actual y-value and predicted y-value
// Parameters    : y_i {std::vector<double>} - Vector of actual y-values, y_hat_i {std::vector<double>} - Vector of predicted y-values
// Return values : {double} - Mean loss of the function given the y_i and y_hat_i parameters
double ML::LogisticRegression::error(std::vector<double> y_i, std::vector<double> y_hat_i){
  size_t len = y_i.size();
  double loss = 0.0;

  for(int i = 0; i < len; i++){
    loss += logloss(y_i[i], y_hat_i[i]);
  }
  
  return ((1.0 / len) * loss);
}



// Purpose       : Find the gradients of the intercept and weights by calculating the partial derivative with respect to x_ij for each element in the beta vector
// Parameters    : X {std::vector<std::vector<double>>} - Input matrix, b_i {std::vector<double>} - Current vector of intercept and weights, y_i {std::vector<double>} - Vector of actual y-values
// Return values : {std::vector<double>} - Vector of gradients corresponding to the order of elements of the beta parameter
std::vector<double> ML::LogisticRegression::gradient_cost(std::vector< std::vector<double> > X, std::vector<double> b_i, std::vector<double> y_i){
  int size = b_i.size();
  std::vector<double> cost(size, 0.0);

  for(int i = 0; i < X.size(); i++){
    double err = predict(X[i], b_i) - y_i[i];

    for(int j = 0; j < X[0].size(); j++){
      cost.at(j) += (err * X[i][j]);
    }

  } // end of nested loop

  for(int i = 0; i < cost.size(); i++){
    cost[i] = (1.0 / size) * cost[i];
  }

  return cost;
}



// Purpose       : Piece together previously defined methods to perform gradient descent in order to find optimal intercept and weights
// Parameters    : X {std::vector<std::vector<double>>} - Input matrix, b_i {std::vector<double>} - Initial random intercept and weights, y_i {std::vector<double>} - Vector of actual y-values
// Return values : {std::vector<double>} - Updated beta parameter (best fit intercept and weights)
std::vector<double> ML::LogisticRegression::gradient_descent(const std::vector< std::vector<double> > X, std::vector<double> b_i, const std::vector<double> y_i){
  std::vector<double> b_i_s = b_i;

  for(int epoch = 0; epoch < epochs; epoch++){
    std::vector<double> y_hat_i(y_i.size(), 0.0);

    for(int i = 0; i < X.size(); i++){
      y_hat_i[i] = predict(X[i], b_i_s);
    }

    // Log the error per 1000 epochs, can be commented out
    /*
    if(epoch % 500 == 0){
      double loss = error(y_i, y_hat_i);
      printf("Error # at epoch #%d: %f\n", epoch, loss);
    }
    */

    // Calculate the gradients of the beta vector
    std::vector<double> cost = gradient_cost(X, b_i_s, y_i);

    // Step down - actual gradient descent for each element in the updated beta list
    for(int j = 0; j < cost.size(); j++){
      b_i_s[j] = b_i_s[j] - (cost[j] * learning_rate);
    }
  }

  return b_i_s;
  
}
