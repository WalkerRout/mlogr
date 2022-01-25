#include "wml.h"
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <algorithm>

// ***************
// Purpose       :
// Parameters    :
// Return values :
// ***************



// Purpose       :
// Parameters    :
// Return values :
void ML::printMat(std::vector< std::vector<double> > mat){
  for(int i = 0; i < mat.size(); i++){
    for(int j = 0; j < mat[i].size(); j++){
      printf("%f ", mat[i][j]);
    }
    printf("\n");
  }
}



// Purpose       :
// Parameters    :
// Return values :
void ML::printVec(std::vector<double> vec){
  for(int i = 0; i < vec.size(); i++){
    printf("%f ", vec[i]);
  }
  printf("\n");
}



// Purpose       :
// Parameters    :
// Return values :
std::vector< std::vector<double> > ML::transpose(std::vector< std::vector<double> > mat){
  std::vector< std::vector<double> > mat_n(mat[0].size(), std::vector<double>(mat.size(), 0.0));
  
  for(int i = 0; i < mat.size(); i++){
    for (int j = 0; j < mat[i].size(); j++){
      mat_n[j][i] = mat[i][j];
    }
  }

  return mat_n;
}



// Purpose       :
// Parameters    :
// Return values :
double ML::mean(std::vector<double> x_i){
  double sum = 0.0;

  for(auto x_ij : x_i){
    sum += x_ij;
  }

  return sum / x_i.size();
}



// Purpose       :
// Parameters    :
// Return values :
double ML::std_dev(std::vector<double> x_i){
  size_t num_xs = x_i.size();
  double mn = mean(x_i);
  double sum_diff = 0.0;

  for(auto x_ij : x_i){
    sum_diff += pow((x_ij - mn), 2.0);
  }

  return sqrt(1.0 / (num_xs - 1.0) * sum_diff);
}



// Purpose       :
// Parameters    :
// Return values :
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



// Purpose       :
// Parameters    :
// Return values :
void ML::fill_rand(std::vector<double> &vec){
  srand(time(0));
  std::generate(vec.begin(), vec.end(), rand);
  std::for_each(vec.begin(), vec.end(), [] (double &ele){ele /= RAND_MAX;} );
}



// Class method initializations

// Purpose       :
// Parameters    :
// Return values :
ML::LogisticRegression::LogisticRegression(unsigned int epochs, double learning_rate){
  this->epochs = epochs;
  this->learning_rate = learning_rate;
}




// Purpose       :
// Parameters    :
// Return values :
double ML::LogisticRegression::sigmoid(double z){
  return 1.0 / (1.0 + exp(-z));
}



// Purpose       :
// Parameters    :
// Return values :
double ML::LogisticRegression::dot(std::vector<double> x_i, std::vector<double> b_i){
  double prod = 0.0;

  for(int i = 0; i < x_i.size(); i++){
    double x_ij = x_i[i];
    double b = b_i[i];
    prod += x_ij*b;
  }
  
  return prod;
}



// Purpose       :
// Parameters    :
// Return values :
double ML::LogisticRegression::squish(std::vector<double> x_i, std::vector<double> b_i){
  return sigmoid(dot(x_i, b_i));
}



// Purpose       :
// Parameters    :
// Return values :
double ML::LogisticRegression::logloss(double y_i, double y_hat){
  return -((y_i * log(y_hat)) + ((1.0 - y_i) * log(1.0 - y_hat)));
}



// Purpose       :
// Parameters    :
// Return values :
double ML::LogisticRegression::error(std::vector<double> y_i, std::vector<double> y_hat_i){
  size_t len = y_i.size();
  double loss = 0.0;

  for(int i = 0; i < len; i++){
    loss += logloss(y_i[i], y_hat_i[i]);
  }
  
  return ((1.0 / len) * loss);
}



// Purpose       :
// Parameters    :
// Return values :
std::vector<double> ML::LogisticRegression::gradient_cost(std::vector< std::vector<double> > X, std::vector<double> b_i, std::vector<double> y_i){
  int size = b_i.size();
  std::vector<double> cost(size, 0.0);

  for(int i = 0; i < X.size(); i++){
    double err = squish(X[i], b_i) - y_i[i];

    for(int j = 0; j < X[0].size(); j++){
      cost.at(j) += (err * X[i][j]);
    }

  } // end of nested loop

  for(int i = 0; i < cost.size(); i++){
    cost[i] = (1.0 / size) * cost[i];
  }

  return cost;
}



// Purpose       :
// Parameters    :
// Return values :
std::vector<double> ML::LogisticRegression::gradient_descent(const std::vector< std::vector<double> > X, std::vector<double> b_i, const std::vector<double> y_i){
  std::vector<double> b_i_s = b_i;

  for(int epoch = 0; epoch < epochs; epoch++){
    std::vector<double> y_hat_i(y_i.size(), 0.0);

    for(int i = 0; i < X.size(); i++){
      y_hat_i[i] = squish(X[i], b_i_s);
    }

    if(epoch % 1000 == 0){
      double loss = error(y_i, y_hat_i);
      printf("Error # at epoch #%d: %f\n", epoch, loss);
    }

    std::vector<double> cost = gradient_cost(X, b_i_s, y_i);

    // Step down - actual gradient descent for each element in the updated beta list
    for(int j = 0; j < cost.size(); j++){
      b_i_s[j] = b_i_s[j] - (cost[j] * learning_rate);
    }
  }

  return b_i_s;
  
}