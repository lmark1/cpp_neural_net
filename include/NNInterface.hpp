#ifndef NNINTERFACE_HPP
#define NNINTERFACE_HPP

#include <vector>
#include <eigen3/Eigen/Dense>

/**
 * @brief This is an abstract class meant to be implemented with neural network
 * functionalities and features.
 *
 * @tparam Sample Dataset sample type.
 * @tparam Label Dataset label type.
 */
class NNInterface
{
public:
  NNInterface() = delete;
  NNInterface(const NNInterface &) = default;
  NNInterface(NNInterface &&) = default;
  NNInterface &operator=(const NNInterface &) = default;
  NNInterface &operator=(NNInterface &&) = default;
  virtual ~NNInterface() = 0;

  /**
   * @brief Find the neural network weights that minimize the loss with respect to the
   * given dataset.
   *
   * @param samples Datatset samples.
   * @param labels Dataset labels.
   */
  virtual void train(const std::vector<Eigen::VectorXd> &samples,
    const std::vector<Eigen::VectorXd> &labels) = 0;

  /**
   * @brief Predict
   *
   * @param sample Single dataset sample.
   * @param label Single dataset label.
   */
  virtual double predict(const Eigen::VectorXd &sample, Eigen::VectorXd label) = 0;

  std::vector<double> get_weights() const { return m_weights; };
  void set_weights(std::vector<double> weights) { m_weights = std::move(weights); }

private:
  std::vector<double> m_weights;
};

#endif /* NNINTERFACE_HPP */