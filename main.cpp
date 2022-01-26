#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>
#include <typeinfo>
#include "wml/wml.h"




int main() {
  // Compile: g++ -std=c++11 -w *.cpp wml/*.cpp -o bin/log

  std::vector< std::vector<double> > X = ML::readCSV("data/framingham_clean.csv");
  std::vector<double> y_i;
  ML::splitVariables(X, y_i);

  std::vector<double> b_i(X[0].size(), 0.0);
  ML::fill_rand(b_i);
  X = ML::zscore(X);

  ML::LogisticRegression logr = ML::LogisticRegression(3001, 0.01);
  std::vector<double> updated_beta = logr.gradient_descent(X, b_i, y_i);
  ML::printVec(updated_beta);

  double total = y_i.size();
  ML::accuracy(logr, X, y_i, updated_beta, total, 0.5);

}