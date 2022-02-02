#include <iostream>
#include <stdio.h>
#include <fstream>
#include "wml/wml.h"



void init(std::vector< std::vector<double> > &X, std::vector<double> &y_i, std::vector<double> &b_i, std::string path){
  X = ML::readCSV(path);
  ML::splitVariables(X, y_i);
  b_i.resize(X[0].size(), 0.0);
  ML::fill_rand(b_i);
  X = ML::zscore(X);
}



int main() {
  // Compile: g++ -std=c++11 -w *.cpp wml/*.cpp -o bin/log
  // https://stackoverflow.com/questions/35953886/include-path-directory
  // https://gcc.gnu.org/onlinedocs/cpp/Search-Path.html

  std::vector< std::vector<double> > X;
  std::vector<double> y_i;
  std::vector<double> b_i;
  std::string path = "data/CHD/framingham_clean.csv";

  init(X, y_i, b_i, path);
  
  ML::LogisticRegression logr = ML::LogisticRegression(1001, 0.001);
  std::vector<double> updated_beta = logr.gradient_descent(X, b_i, y_i);
  ML::printVec(updated_beta);

  ML::accuracy(logr, X, y_i, updated_beta, 0.50);
  
  // First line in the cleaned dataset is the person to predict
  /*
  std::cout << std::endl;
  double y_hat = logr.predict(X[0], updated_beta);
  printf("%f percent\n", y_hat);
  */
}
