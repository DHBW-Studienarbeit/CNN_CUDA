/*
 * Layer.hpp
 *
 *  Created on: 23.11.2017
 *      Author: Benjamin Riedle
 */

#ifndef LAYER_HPP_
#define LAYER_HPP_


typedef enum {
	INPUT_LAYER, CONV_LAYER, POOLING_LAYER, FULLY_CONNECTED_LAYER,
	DROPOUT_LAYER
} LAYER_TYPE;

class Layer
{
public:

	LAYER_TYPE type; /* type of this layer */
	int no_nodes; /* combined number of nodes for this layer */
	int node_index;/*index for first Matrix of this Layer in node_list*/
	int bias_index;/*index for first Matrix of this Layer in bias_list*/
	int weight_index;/*index for first Matrix of this Layer in weight_list*/

public:



	Layer(int size, LAYER_TYPE layer_type);
	Layer(LAYER_TYPE layer_type);
	virtual ~Layer();

	LAYER_TYPE getLayerType();
	int getSize();
	void setSize(int new_size);
	int getNodeIndex();
	int getWeightIndex();
	int getBiasIndex();
	void setNodeIndex(int index);
	void setWeightIndex(int index);
	void setBiasIndex(int index);

};

typedef struct{
	LAYER_TYPE type;
	int x_size;
	int y_size;
	int x_receptive;
	int y_receptive;
	int no_feature_maps;
	int step_size;
	int size;
}LAYER_STRUCT;



#endif /* LAYER_HPP_ */
