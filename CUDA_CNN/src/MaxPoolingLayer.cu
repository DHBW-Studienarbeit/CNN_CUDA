/*
 * MaxPoolingLayer.cpp
 *
 *  Created on: 29.11.2017
 *      Author: Benjamin Riedle
 */

#include "MaxPoolingLayer.h"

MaxPooling_Layer::MaxPooling_Layer(int x_receptive, int y_receptive, int no_features) : Layer(POOLING_LAYER) {
	this->x_receptive = x_receptive;
	this->y_receptive = y_receptive;
	this->no_features = no_features;

	this->x_size = 0;
	this->y_size = 0;
}

MaxPooling_Layer::~MaxPooling_Layer() {
	/** nothing to destroy */
}

int MaxPooling_Layer::getNoFeatures()
{
	return no_features;
}

int MaxPooling_Layer::getXSize()
{
	return x_size;
}

int MaxPooling_Layer::getYSize()
{
	return y_size;
}

int MaxPooling_Layer::getXReceptive()
{
	return x_receptive;
}

int MaxPooling_Layer::getYReceptive()
{
	return y_receptive;
}

void MaxPooling_Layer::setXSize(int size)
{
	this->x_size = size;
}

void MaxPooling_Layer::setYSize(int size)
{
	this->y_size = size;
}

