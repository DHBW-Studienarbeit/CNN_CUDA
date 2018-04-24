/*
 * PictureContainer.cpp
 *
 *  Created on: 03.12.2017
 *      Author: Florian
 */

#include "PictureContainer.h"
#include <fstream>
#include <sstream>



PictureContainer::PictureContainer(std::string foldername, int num_of_files)
{
	this->next_index = -1;
	this->file_index = 0;
	this->foldername = foldername;
	this->num_of_files = num_of_files;
	for(int i = 0; i < PICS_PER_FILE; i++)
	{
		images[i] = Picture();
	}
	load_pictures();
}

PictureContainer::~PictureContainer() {

}

void PictureContainer::load_pictures() {
	std::ostringstream ss;
	ss << this->file_index;
	std::string csv_file = this->foldername + "/" + ss.str() + ".csv";
	std::ifstream infile(csv_file.c_str());
	for(int i=0; i<PICS_PER_FILE; i++)
	{
		std::string line;
		std::getline(infile,line);
		this->images[i] = Picture(&line);
	}
}

Picture * PictureContainer::get_nextpicture(void)
{
	next_index++;
	if(next_index >= PICS_PER_FILE)
	{
		next_index=0;
		file_index++;
		if(file_index > num_of_files)
		{
			file_index = 0;
		}
		load_pictures();
	}
	return this->images + next_index;
}
