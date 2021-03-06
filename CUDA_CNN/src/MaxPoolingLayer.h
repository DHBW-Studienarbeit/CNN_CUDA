/*
 * MaxPoolingLayer.hpp
 *
 *  Created on: 29.11.2017
 *      Author: Benjamin Riedle
 */

#ifndef MAXPOOLINGLAYER_HPP_
#define MAXPOOLINGLAYER_HPP_

#include "Layer.h"

class MaxPooling_Layer: public Layer {
public:
	int x_size;
	int y_size;
	int x_receptive;
	int y_receptive;
	int no_features;
public:
	MaxPooling_Layer(int x_receptive, int y_receptive, int no_features);
	virtual ~MaxPooling_Layer();

	int  getNoFeatures();
	int  getXSize();
	int  getYSize();
	void setXSize(int size);
	void setYSize(int size);
	int  getXReceptive();
	int  getYReceptive();

};

#endif /* MAXPOOLINGLAYER_HPP_ */
