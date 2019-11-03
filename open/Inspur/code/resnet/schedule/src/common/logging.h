/*
 *\logger.h
 *\brief logging module
 */
#pragma once
#include <map>
#include <vector>
#include <mutex>
#include <memory>
#include <time.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdint>

#include "types.h"
#include "string_op.h"


using namespace std;


namespace common {

	///
	/// \brief log level
	///
	typedef enum class log_level {
		LOG_INFO,
		LOG_WARNING,
		LOG_ERROR,
		LOG_FATAL
	} log_level_t;

	///
	/// \brief log config
	///
	typedef struct log_conf {
		log_conf(string f, ofstream* s, mutex* m) {
			file = f;
			stream = s;
			mtx = m;
		}
		string file;
		ofstream* stream;
		mutex* mtx;
	} log_conf_t;

	struct LogBinaryAsHexString {
		vector<uint8_t>* data;
	};

	const string& ArgValueTransform(const bool& value);
	const string ArgValueTransform(const LogBinaryAsHexString& value);

	template <typename T>
	const T& ArgValueTransform(const T& value) {
		return value;
	}

	///
	/// \brief log system class
	///
	class Logger {
	public:
		~Logger();

		static Logger* GetLogger(const string& info_log_filename = "mlperf_schedule_info.log",
			const string& warn_log_filename = "mlperf_schedule_warn.log",
			const string& error_log_filename = "mlperf_schedule_error.log");

		template <typename... Args>
		static void Log(Args... args) {
			GetLogger()->_Log(log_level_t::LOG_INFO, args...);
		};

		template <typename... Args>
		static void LogDuration(common::ScheduleClock::duration duration, Args... args) {
			GetLogger()->_LogDuration(log_level_t::LOG_INFO, duration, args...);
		};

		template <typename... Args>
		static void LogInfo(Args... args) {
			GetLogger()->_Log(log_level_t::LOG_INFO, args...);
		};

		template <typename... Args>
		static void LogWarning(Args... args) {
			GetLogger()->_Log(log_level_t::LOG_WARNING, args...);
		};

		template <typename... Args>
		static void LogError(Args... args) {
			GetLogger()->_Log(log_level_t::LOG_ERROR, args...);
		};

	private:
		//construct
		Logger(const string& info_log_filename,
			const string& warn_log_filename,
			const string& erro_log_filename){
			m_info_log_file = move(info_log_filename);
			m_warn_log_file = move(warn_log_filename);
			m_error_log_file = move(erro_log_filename);
			m_info_stream.open(m_info_log_file.c_str());
			m_warn_stream.open(m_warn_log_file.c_str());
			m_error_stream.open(m_error_log_file.c_str());
			m_level_conf_map[log_level_t::LOG_INFO] = make_shared<log_conf_t>(m_info_log_file, &m_info_stream, &m_info_mutex);
			m_level_conf_map[log_level_t::LOG_WARNING] = make_shared<log_conf_t>(m_warn_log_file, &m_warn_stream, &m_warn_mutex);
			m_level_conf_map[log_level_t::LOG_ERROR] = make_shared<log_conf_t>(m_error_log_file, &m_error_stream, &m_error_mutex);
		};
		Logger(Logger&) = delete;
		Logger(Logger&&) = delete;
		Logger& operator=(Logger&) = delete;
		Logger& operator=(Logger&&) = delete;

		template <typename... Args>
		void _Log(
			log_level_t log_level,
			Args... args) {
			auto timestamp = common::ScheduleClock::now();
			ostream& stream = GetStream(log_level);

			unique_lock<mutex> lock(*m_level_conf_map[log_level]->mtx);
//#ifdef WIN32
//			char str[50];
//			ctime_s(str, sizeof str, &now);
//			stream << "[" << common::strip(str) << "] ";
//#else
//			stream << "[" << common::strip(ctime(&now)) << "] ";
//#endif

			stream << "[" << timestamp.time_since_epoch().count() << "] ";
			LogArgs(&stream, args...);
			stream << endl;
			stream << flush;
			lock.unlock();
		};

		template <typename... Args>
		void _LogDuration(
			log_level_t log_level,
			common::ScheduleClock::duration duration,
			Args... args) {
			ostream& stream = GetStream(log_level);

			unique_lock<mutex> lock(*m_level_conf_map[log_level]->mtx);

			static char dur_s[100];
			sprintf(dur_s, "%.5f", static_cast<double>(duration.count()) / 1000000.0);

			stream << "[" << dur_s << "] ";
			LogArgs(&stream, args...);
			stream << endl;
			stream << flush;
			lock.unlock();
		};

		///
		/// \brief get output stream by log level
		///
		ostream& GetStream(log_level_t log_level) { return *m_level_conf_map[log_level]->stream; };

		void LogArgs(ostream*) {};

		template <typename T>
		void LogArgs(ostream* out, const T& value_only) {
			*out << ArgValueTransform(value_only);
		};

		template <typename T>
		void LogArgs(ostream* out, const string& arg_name,
			const T& arg_value) {
			*out << "\"" << arg_name << "\" : " << ArgValueTransform(arg_value);
		};
		
		template <typename T, typename... Args>
		void LogArgs(ostream* out, const string& arg_name,
			const T& arg_value, const Args... args) {
			*out << "\"" << arg_name << "\" : " << ArgValueTransform(arg_value) << ", ";
			LogArgs(out, args...);
		};

		static Logger* m_p_logger;
		static mutex m_log_mutex;

		map<log_level_t, shared_ptr<log_conf_t>> m_level_conf_map;

		string m_info_log_file;
		string m_warn_log_file;
		string m_error_log_file;

		ofstream m_info_stream;
		ofstream m_warn_stream;
		ofstream m_error_stream;

		mutex m_info_mutex;
		mutex m_warn_mutex;
		mutex m_error_mutex;
	};

}
