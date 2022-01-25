#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>
#include "wml.h"


std::vector<std::vector<double>> readCSV(std::string filename){
  std::vector<std::vector<double>> mat;

  std::ifstream fin(filename);
  std::string linestr;

  while(std::getline(fin, linestr)){
    std::stringstream ss(linestr);
    std::vector<double> temp;
    std::string data;

    while(std::getline(ss, data, ',')){
      temp.push_back(std::stod(data));
    }
    if (temp.size() > 0) mat.push_back(temp);
  }

  return mat;
}



void splitVariables(std::vector<std::vector<double>> &X, std::vector<double> &y_i){
  for(auto &x_i : X){
    y_i.push_back(x_i[x_i.size() - 1]);
    x_i.pop_back();
    x_i.insert(x_i.begin(), 1); // Easier dot product calculations after appending a 1 to the beginning
  }
}



int main() {
  std::vector<std::vector<double>> X = readCSV("marks.csv");
  std::vector<double> y_i;
  splitVariables(X, y_i);

  std::vector<double> b_i(X[0].size(), 0.0);
  ML::fill_rand(b_i);
  X = ML::zscore(X);

  ML::LogisticRegression logr = ML::LogisticRegression(15001, 0.008);
  std::vector<double> updated_beta = logr.gradient_descent(X, b_i, y_i);
  ML::printVec(updated_beta);

}