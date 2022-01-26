
// Define header
#ifndef WML_H
#define WML_H


/* Include Statements */
/* ------------------ */

#include <vector>

/* ------------------ */



// Namespace for machine learning models
namespace ML {
  class LogisticRegression {
    private:
      unsigned int epochs;
      double learning_rate;

      double sigmoid(const double z);
      double dot(const std::vector<double> x_i, const std::vector<double> b_i);
      double logloss(const double y, const double y_hat);
      double error(const std::vector<double> y_i, const std::vector<double> y_hat_i);
      std::vector<double> gradient_cost(const std::vector< std::vector<double> > X, const std::vector<double> b_i, const std::vector<double> y_i);

    public:
      LogisticRegression(const unsigned int epochs, const double learning_rate);
      std::vector<double> gradient_descent(const std::vector< std::vector<double> > X, std::vector<double> b_i, const std::vector<double> y_i);
      double predict(const std::vector<double> x_i, const std::vector<double> b_i);
  };

  double mean(std::vector<double> x_i);
  double std_dev(std::vector<double> x_i);

  std::vector< std::vector<double> > zscore(std::vector< std::vector<double> > X);
  std::vector< std::vector<double> > transpose(std::vector< std::vector<double> > mat);
  std::vector< std::vector<double> > readCSV(std::string filename);

  void splitVariables(std::vector< std::vector<double> > &X, std::vector<double> &y_i);
  void accuracy(ML::LogisticRegression logr, std::vector< std::vector<double> > X, std::vector<double> y_i, std::vector<double> b_i, double total, double thresh);
  void printMat(std::vector< std::vector<double> > mat);
  void printVec(std::vector<double> vec);
  void fill_rand(std::vector<double> &vec);
  
}

#endif