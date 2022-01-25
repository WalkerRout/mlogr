#ifndef WML_H
#define WML_H

#include <vector>

namespace ML {
  class LogisticRegression {
    private:
      unsigned int epochs;
      double learning_rate;

      double sigmoid(const double z);
      double dot(const std::vector<double> x_i, const std::vector<double> b_i);
      double squish(const std::vector<double> x_i, const std::vector<double> b_i);
      double logloss(const double y, const double y_hat);
      double error(const std::vector<double> y_i, const std::vector<double> y_hat_i);
      std::vector<double> gradient_cost(const std::vector<std::vector<double>> X, const std::vector<double> b_i, const std::vector<double> y_i);

    public:
      LogisticRegression(const unsigned int epochs, const double learning_rate);
      std::vector<double> gradient_descent(const std::vector<std::vector<double>> X, std::vector<double> b_i, const std::vector<double> y_i);
  };

  double mean(std::vector<double> x_i);
  double std_dev(std::vector<double> x_i);
  std::vector<std::vector<double>> zscore(std::vector<std::vector<double>> X);
  std::vector<std::vector<double>> transpose(std::vector<std::vector<double>> mat);
  void printMat(std::vector<std::vector<double>> mat);
  void printVec(std::vector<double> vec);
  void fill_rand(std::vector<double> &vec);
  
}

#endif