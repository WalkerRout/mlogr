#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>
#include "wml.h"


std::vector< std::vector<double> > readCSV(std::string filename){
  std::vector< std::vector<double> > mat;

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



void splitVariables(std::vector< std::vector<double> > &X, std::vector<double> &y_i){
  for(auto &x_i : X){
    y_i.push_back(x_i[x_i.size() - 1]);
    x_i.pop_back();
    x_i.insert(x_i.begin(), 1); // Easier dot product calculations after appending a 1 to the beginning
  }
}



void accuracy(ML::LogisticRegression logr, std::vector< std::vector<double> > X, std::vector<double> y_i, std::vector<double> b_i, double total, double thresh){
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

  printf("Accuracy: %.2f percent!\n", (((tpos + tneg) / total) * 100));
}



int main() {
  // Compile: g++ -std=c++11 -w *.cpp -o log

  std::vector< std::vector<double> > X = readCSV("marks.csv");
  std::vector<double> y_i;
  splitVariables(X, y_i);

  std::vector<double> b_i(X[0].size(), 0.0);
  ML::fill_rand(b_i);
  X = ML::zscore(X);

  ML::LogisticRegression logr = ML::LogisticRegression(3001, 0.007);
  std::vector<double> updated_beta = logr.gradient_descent(X, b_i, y_i);
  ML::printVec(updated_beta);

  double total = y_i.size();
  accuracy(logr, X, y_i, updated_beta, total, 0.5);

}