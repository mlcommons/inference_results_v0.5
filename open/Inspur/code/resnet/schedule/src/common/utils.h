#pragma once
#include <map>
#include <string>
#include <chrono>
#include <stdarg.h>
#ifdef WIN32
#include <io.h>
#include <windows.h>
#include <direct.h>
#else
#include <unistd.h>
#include <dirent.h>
#include <linux/limits.h>
#include <sys/types.h>
#include <sys/stat.h>
#endif

#ifdef WIN32
constexpr char PATH_SEP = '\\';
#define PATH_ACCESS(path) _access(path.c_str(), 0)
#define DIR_MAKE(dir) _mkdir(dir.c_str())
#define DIR_REMOVE(dir) _rmdir(dir.c_str())
#else
constexpr char PATH_SEP = '/';
#define PATH_ACCESS(path) access(path.c_str(), 0)
#define DIR_MAKE(dir) mkdir(dir.c_str(), 0777)
#define DIR_REMOVE(dir) rmdir(dir.c_str())
#endif

using namespace std;


namespace common {

	class Utils {
	public:
		static string GetExeDir() {
#ifdef WIN32
			TCHAR exepath[MAX_PATH + 1];
			GetModuleFileName(NULL, exepath, MAX_PATH);
			string path = string(exepath);
#else
			char arg1[20];
			char exepath[PATH_MAX + 1] = { 0 };
			sprintf(arg1, "/proc/%d/exe", getpid());
			ssize_t _ = readlink(arg1, exepath, 1024);
			string path = string(exepath);
#endif
			string::size_type position = path.rfind(PATH_SEP);
			return path.substr(0, position + 1);
		};

		static string PathJoin(size_t count, ...) {
			string result;
			va_list pvar;
			va_start(pvar, count);
			for (size_t i = 0; i < count; ++i) {
				string t = va_arg(pvar, const char*);
				if (t == "")
					continue;
				if (t[t.length() - 1] != PATH_SEP && i != count - 1) {
					t = t + PATH_SEP;
				}
				result += t;
			}
			va_end(pvar);
			return result;
		};

		static bool MakeDir(string dir) {
			if (dir == "")
				return true;
			if (PATH_ACCESS(dir) == -1)
			{
				int flag = DIR_MAKE(dir);
				if (flag == 0) {
					return true;
				}
				else {
					return false;
				}
			}
			return true;
		};

		static bool RemoveDir(string dir) {
			if (dir == "")
				return true;
			if (PATH_ACCESS(dir) == 0)
			{
				int flag = DIR_REMOVE(dir);
				if (flag == 0)
				{
					return true;
				}
				else {
					return false;
				}
			}
			return true;
		};

		static string GetDir(string path) {
			if (path == "")
				return path;
			string::size_type position = path.rfind(PATH_SEP);
			if (position == path.length() - 1)
				return path;
			return path.substr(0, position + 1);
		};

		static bool PathExist(string path) {
			if (path == "")
				return false;
			if (PATH_ACCESS(path) == 0)
				return true;
			else
				return false;
		};

		static void ChangePathSep(string path, string& new_path, string& sep) {
			string path_sep_old;
			string path_sep_new;
			while (true) {
				string::size_type pos(0);
				if ((pos = path.find(path_sep_old)) != string::npos)
					path = path.replace(pos, path_sep_old.length(), path_sep_new);
				else
					break;
			}
			new_path = path;
			sep = path_sep_new;
		}
	};

}
