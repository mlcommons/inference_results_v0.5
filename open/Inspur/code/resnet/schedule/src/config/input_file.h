#ifndef __INPUT_FILE_H__
#define __INPUT_FILE_H__
#include <string>
#include <vector>
#include <fstream>

using namespace std;


namespace schedule {

	class InputFile {
	public:
		InputFile(string path);
		~InputFile();

		string ReadLine();
		streampos TellPos();
		void SeekPos(streampos pos);
		string GetPath() const;
		ifstream& GetStream();
	private:
		string m_path;
		ifstream m_stream;
	};

}


#endif // !__INPUT_FILE_H__
