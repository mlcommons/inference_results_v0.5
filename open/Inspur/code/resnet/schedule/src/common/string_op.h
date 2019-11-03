#ifndef __STRING_OP_H__
#define __STRING_OP_H__
#include <string>

#include "common.h"

using namespace std;


namespace common {

	string do_strip(const string& str, StringStripType striptype, const string& chars);
	string strip(const string& str, const string& chars = "");
	string lstrip(const string& str, const string& chars = "");
	string rstrip(const string& str, const string& chars = "");

}

#endif // !__STRING_OP_H__