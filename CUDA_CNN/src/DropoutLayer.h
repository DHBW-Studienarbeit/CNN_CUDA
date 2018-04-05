/*
 * DropoutLayer.hpp
 *
 *  Created on: 29.11.2017
 *      Author: Benjamin Riedle
 */

#ifndef DROPOUTLAYER_HPP_
#define DROPOUTLAYER_HPP_

#include "Layer.h"

class Dropout_Layer: public Layer {
public:
	Dropout_Layer();
	virtual ~Dropout_Layer();

};

#endif /* DROPOUTLAYER_HPP_ */
