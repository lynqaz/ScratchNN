#pragma once

#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>


class AI
{
public:
	AI(std::vector<int> extent);
	AI(std::vector<std::vector<std::vector<double>>> weights, std::vector<std::vector<std::vector<double>>> biases);
	void train(std::vector<std::vector<double>> training_inputs, std::vector<std::vector<double>> training_outputs, std::vector<std::vector<double>> test_inputs, std::vector<std::vector<double>> test_outputs, int iterations, int batchsize, double learningrate);
	void getparams(std::vector<std::vector<std::vector<double>>>& weights, std::vector<std::vector<std::vector<double>>>& biases);
	std::vector<double> run(std::vector<double> input);
	double getloss(std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> expected_outputs);
private:
	enum class parameter { weight, bias };
	std::vector<int> extent;
	std::vector<std::vector<std::vector<double>>> weights;
	std::vector<std::vector<std::vector<double>>> biases;

	std::vector<double> feedforward(std::vector<double> input, std::vector<std::vector<double>>& activations, std::vector<std::vector<double>>& weightedInputs);
	void backpropagation(std::vector<double> input, std::vector<double> output, std::vector<std::vector<std::vector<double>>>& deltaW, std::vector<std::vector<std::vector<double>>>& deltaB);
	void updateparams(std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> outputs, double learningrate);
	double evaluate(std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> expected_outputs);
	void MatrixGenerator(double f(), std::vector<std::vector<std::vector<double>>>& matrix, parameter p);

	static double zero();
	static double normalDistribution();
	static double activationFunction(double);
	static double activationFunction_derivative(double);
	static double costfunction(double, double);
	static double costfunction_derivative(double, double);
};

