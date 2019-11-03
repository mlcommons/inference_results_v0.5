#include <string.h>

#include "common.h"
#include "string_op.h"

using namespace std;


namespace common {

	string do_strip(const string& str, StringStripType striptype, const string& chars)
	{
		size_t strlen = str.size();
		size_t charslen = chars.size();
		size_t i = 0, j = strlen - 1;

		if (0 == charslen)
		{
			if (striptype != StringStripType::RIGHTSTRIP)
			{
				while (i < strlen && ::isspace(str[i]))
				{
					i++;
				}
			}
			if (striptype != StringStripType::LEFTSTRIP)
			{
				while (j > i && ::isspace(str[j]))
				{
					j--;
				}
			}
		}
		else
		{
			const char* sep = chars.c_str();
			if (striptype != StringStripType::RIGHTSTRIP)
			{
				while (i < strlen && memchr(sep, str[i], charslen))
				{
					i += charslen;
				}
			}
			j = strlen - charslen;
			if (striptype != StringStripType::LEFTSTRIP)
			{
				while (j > i && memchr(sep, str[j], charslen))
				{
					j -= charslen;
				}
			}
		}
		if (i >= 0 && i <= j && j <= strlen - 1)
		{
			return str.substr(i, j - i + 1);
		}
		else
		{
			return "";
		}
	}

	string strip(const string& str, const string& chars)
	{
		return do_strip(str, StringStripType::BOTHSTRIP, chars);
	}

	string lstrip(const string& str, const string& chars)
	{
		return do_strip(str, StringStripType::LEFTSTRIP, chars);
	}

	string rstrip(const string& str, const string& chars)
	{
		return do_strip(str, StringStripType::RIGHTSTRIP, chars);
	}

}