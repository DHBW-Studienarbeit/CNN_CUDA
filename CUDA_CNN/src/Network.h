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
#define BATCH_SIZE 150

using namespace std;

class Network {

private:
	vector<Layer*>* layer_list;

	Layer* device_layer_list; /* stored as array at GPU memory*/

	/** Arrays of pointer to allocated memory for matrices on GPU memory  */
	float **nodeArrayPtrs;
	float **weightArrayPtrs;
	float **biasArrayPtrs;
	float **nodeDerivArrayPtrs;
	float **weightDerivArrayPtrs;
	float **biasDerivArrayPtrs;

	float **nodeDeviceArrayPtrs;
	float **weightDeviceArrayPtrs;
	float **biasDeviceArrayPtrs;
	float **nodeDerivDeviceArrayPtrs;
	float **weightDerivDeviceArrayPtrs;
	float **biasDerivDeviceArrayPtrs;

	/* Arrays of array lengths */
	int *nodeArrayLengths;
	int *weightArrayLengths;
	int *biasArrayLengths;

	/* X and Y dimensions of arrays *
	 * ROWS (X)  COLS (Y)			*/
	int *nodeMatrixDims_x;
	int *weightMatrixDims_x;
	int *biasMatrixDims_x;

	int *nodeMatrixDims_y;
	int *weightMatrixDims_y;
	int *biasMatrixDims_y;

	/* GPU device memory for array lengths */
	int *nodeDeviceArrayLengths;
	int *weightDeviceArrayLengths;
	int *biasDeviceArrayLengths;
	int *nodeDerivDeviceArrayLengths;
	int *weightDerivDeviceArrayLengths;
	int *biasDerivDeviceArrayLengths;

	/* X and Y dimensions of arrays *
	 * ROWS (X)  COLS (Y) stored at device memory	*/
	int *nodeDeviceMatrixDims_x;
	int *weightDeviceMatrixDims_x;
	int *biasDeviceMatrixDims_x;
	int *nodeDerivDeviceMatrixDims_x;
	int *weightDerivDeviceMatrixDims_x;
	int *biasDerivDeviceMatrixDims_x;

	int *nodeDeviceMatrixDims_y;
	int *weightDeviceMatrixDims_y;
	int *biasDeviceMatrixDims_y;
	int *nodeDerivDeviceMatrixDims_y;
	int *weightDerivDeviceMatrixDims_y;
	int *biasDerivDeviceMatrixDims_y;

	/* matrix counters */
	int no_node_matrices;
	int no_weight_matrices;
	int no_bias_matrices;

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
