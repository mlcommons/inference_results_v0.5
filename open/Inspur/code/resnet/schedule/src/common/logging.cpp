#include <memory>

#include "logging.h"

using namespace std;


namespace common {

	Logger* Logger::m_p_logger = nullptr;
	mutex Logger::m_log_mutex;

	const string& ArgValueTransform(const bool& value) {
		static const string v_true("true");
		static const string v_false("false");
		return value ? v_true : v_false;
	}

	char Bin2Hex(uint8_t four_bits) {
		char number = '0' + four_bits;
		char letter = ('A' - 10) + four_bits;
		return four_bits < 10 ? number : letter;
	}

	const string ArgValueTransform(const LogBinaryAsHexString& value) {
		if (value.data == nullptr) {
			return "\"\"";
		}
		string hex;
		hex.reserve(value.data->size() + 2);
		hex.push_back('"');
		for (auto b : *value.data) {
			hex.push_back(Bin2Hex(b >> 4));
			hex.push_back(Bin2Hex(b & 0x0F));
		}
		hex.push_back('"');
		return hex;
	}

	Logger* Logger::GetLogger(const string& info_log_filename,
		const string& warn_log_filename,
		const string& error_log_filename) {
		if (!m_p_logger) {
			unique_lock<mutex> lock(m_log_mutex);
			if (!m_p_logger) {
				m_p_logger = new Logger(info_log_filename, warn_log_filename, error_log_filename);
			}
			lock.unlock();
		}
		return m_p_logger;
	}

	Logger::~Logger() {
		map<log_level_t, shared_ptr<log_conf_t>>::iterator iter;
		iter = m_level_conf_map.begin();
		while (iter != m_level_conf_map.end()) {
			*iter->second->stream << endl << flush;
			iter->second->stream->close();
			iter++;
		}
	}
}
