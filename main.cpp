#include <iostream>
#include <stdio.h>
#include "wml/wml.h"




int main() {
  // Compile: g++ -std=c++11 -w *.cpp wml/*.cpp -o bin/log
  // https://stackoverflow.com/questions/35953886/include-path-directory

  std::vector< std::vector<double> > X = ML::readCSV("data/framingham_clean.csv");

  std::vector<double> y_i;
  ML::splitVariables(X, y_i);

  std::vector<double> b_i(X[0].size(), 0.0);
  ML::fill_rand(b_i);
  X = ML::zscore(X);

  ML::LogisticRegression logr = ML::LogisticRegression(501, 0.01);
  std::vector<double> updated_beta = logr.gradient_descent(X, b_i, y_i);
  ML::printVec(updated_beta);

  double total = y_i.size();
  ML::accuracy(logr, X, y_i, updated_beta, total, 0.5);
  
  // First line in the cleaned dataset is the person to predict
  /*
  std::cout << std::endl;
  double y_hat = logr.predict(X[0], updated_beta);
  printf("%f percent\n", y_hat);
  */
}