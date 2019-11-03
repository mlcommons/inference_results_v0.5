#include <istream>
#include <cassert>

#include "../common/string_op.h"
#include "input_file.h"

using namespace std;


namespace schedule {

	InputFile::InputFile(string path) {
		m_path = path;
		m_stream = ifstream(m_path);
		assert(m_stream.is_open());
	}

	InputFile::~InputFile() {
		m_stream.clear();
		m_stream.close();
	}

	string InputFile::ReadLine() {
		while (true) {
			char line[1024] = { 0 };
			istream& ret = m_stream.getline(line, sizeof(line));
			if (!ret)
				return "";
			string line_s = line;
			line_s = common::strip(line_s);
			if (line_s == "" || line_s[0] == '#')
				continue;
			return line_s;
		}
	}

	streampos InputFile::TellPos() {
		return m_stream.tellg();
	}

	void InputFile::SeekPos(streampos pos) {
		m_stream.seekg(pos);
	}

	string InputFile::GetPath() const {
		return m_path;
	}

	ifstream& InputFile::GetStream() {
		return m_stream;
	}

}
