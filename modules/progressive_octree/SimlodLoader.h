#pragma once

#include <string>
#include <stdint.h>

using std::string;

void loadFileNative(
	string file, 
	uint64_t firstByte, 
	uint64_t numBytes, 
	void* target,
	uint64_t* out_padding
);


