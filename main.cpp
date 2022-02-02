#include <iostream>
#include <stdio.h>
#include <fstream>

#include <chrono>
#include <thread>
#include <atomic>

#include "wml/wml.h"



void init(std::vector< std::vector<double> > &X, std::vector<double> &y_i, std::vector<double> &b_i, std::string path){
  X = ML::readCSV(path);
  ML::splitVariables(X, y_i);
  b_i.resize(X[0].size(), 0.0);
  ML::fill_rand(b_i);
  X = ML::zscore(X);
}



void wait(std::atomic<bool> &threadDone){

  short amount = 5;
  short counter = 0;

  std::cout << "Waiting" << std::endl;
  while(!threadDone){

    if(counter >= amount){
      printf("\e[1;1H\e[2J");
      counter = 0;
      std::cout << "Waiting" << std::endl;
    }

    std::cout << "." << std::flush;
    
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    counter++;
  }

}



int main() {
  // Compile: g++ -std=c++11 -w -lpthread *.cpp wml/*.cpp -o bin/log
  // https://stackoverflow.com/questions/35953886/include-path-directory
  // https://gcc.gnu.org/onlinedocs/cpp/Search-Path.html

  printf("\e[1;1H\e[2J");

  std::vector< std::vector<double> > X;
  std::vector<double> y_i;
  std::vector<double> b_i;
  std::string path = "data/CHD/framingham_clean.csv";

  init(X, y_i, b_i, path);
  
  ML::LogisticRegression logr = ML::LogisticRegression(1001, 0.001);
  std::vector<double> updated_beta;
  //std::vector<double> updated_beta = logr.gradient_descent(X, b_i, y_i);

  std::atomic<bool> logThreadDone(false);



  std::thread logThread([&](){
    updated_beta = logr.gradient_descent(X, b_i, y_i);
    logThreadDone = true;
  });

  wait(logThreadDone);

  ML::accuracy(logr, X, y_i, updated_beta, 0.50);

  logThread.join();
  
  // First line in the cleaned dataset is the person to predict
  /*
  std::cout << std::endl;
  double y_hat = logr.predict(X[0], updated_beta);
  printf("%f percent\n", y_hat);
  */
}
