/*
 * network.hpp
 *
 *  Created on: 23.11.2017
 *      Author: Benjamin Riedle
 *
 *  This file defines a class Network, which will determine the final
 *  CNN with all used nodes
 *
 */

#ifndef NETWORK_HPP_
#define NETWORK_HPP_

#include <vector>
#include "Layer.h"
#include "matrix.h"
#include "InputLayer.h"
#include "ConvLayer.h"
#include "DropoutLayer.h"
#include "FullyConnectedLayer.h"
#include "MaxPoolingLayer.h"
#include "mathematics.h"
#include "PictureContainer.h"


#define NO_DATA_D	55000
#define NO_TEST_FILES_D	 10000
#define NO_PICS_PER_FILE_D	1000
#define LEARNING_RATE 0.5f

using namespace std;

class Network {

private:
	vector<Layer*>* layer_list;
//	vector<Matrix*>* node_list;
//	vector<Matrix*>* weight_list;
//	vector<Matrix*>* bias_list;
//	vector<Matrix*>* node_deriv_list;
//	vector<Matrix*>* weight_deriv_list;
//	vector<Matrix*>* bias_deriv_list;

	/** Arrays of pointer to allocated memory for matrices on CUDA */
	float **nodeArrayPtrs;
	float **weightArrayPtrs;
	float **biasArrayPtrs;
	float **nodeDerivArrayPtrs;
	float **weightDerivArrayPtrs;
	float **biasDerivArrayPtrs;

	PictureContainer* train_picture_container;
	PictureContainer* test_picture_container;

	void reset_backprop_state(void);

public:
	Network();
	~Network();

	void add_Layer(Layer* layer);
	bool generate_network(); /* returns success of function */
	bool train(int batch_size, int no_iterations); /* returns success of function */
	float test();

private:
	bool backpropagate(float* labels);
	float forward(float* labels);
	void gradient_descent(float cost);

};


#endif /* NETWORK_HPP_ */
