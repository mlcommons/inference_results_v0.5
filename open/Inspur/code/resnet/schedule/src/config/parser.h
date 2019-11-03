#ifndef __PARSER_H__
#define __PARSER_H__
#include <map>
#include <string>
#include <sstream>

#include "../common/string_op.h"
#include "../config/input_file.h"

using namespace std;


namespace schedule {

	class InputFile;
	class Config;
	class Parser {
	public:
		Parser(InputFile* pfile, Config* pconf);
		virtual ~Parser();

		void Run();

		template <typename T>
		void ParseLines(map<string, T*> key_value, string stop_str, bool need_fallback) {
			string line = "";
			stringstream word;
			string key = "";
			T value;
			streampos pos = 0;
			map<string, bool> parsed_map;

			while (true) {
				pos = m_pfile->TellPos();
				line = m_pfile->ReadLine();
				if (line == "") {
					break;
				}
				word = stringstream(line);
				word >> key;
				if (key == stop_str || key_value.count(key) <= 0) {
					break;
				}
				word >> value;
				*key_value[key] = value;
				parsed_map[key] = true;
				if (parsed_map.size() == key_value.size())
					return;
			}
			if (need_fallback) {
				m_pfile->SeekPos(pos);
			}
		}

		template <typename T>
		void ParseLinesV(map<string, vector<T>*> key_value, string stop_str, bool need_fallback) {
			string line = "";
			stringstream word;
			string key = "";
			T value;
			streampos pos = 0;

			while (true) {
				pos = m_pfile->TellPos();
				line = m_pfile->ReadLine();
				if (line == "") {
					break;
				}
				word = stringstream(line);
				word >> key;
				if (key == stop_str || key_value.count(key) <= 0) {
					break;
				}
				word >> value;
				(*key_value[key]).push_back(value);
			}
			if (need_fallback) {
				m_pfile->SeekPos(pos);
			}
		}

	protected:
		InputFile* m_pfile;
		Config* m_pconf;
	};

	class Node;
	class WeightFiller;
	class Include;
	class NodeParser : public Parser {
	public:
		NodeParser(InputFile* pfile, Config* pconf);
		~NodeParser();

		Node Run();
		Node ParseCommon();
		WeightFiller ParseWeightFiller();
		Include ParseInclude();
	};

	class ImageDataNode;
	class ConvolutionNode;
	class PoolingNode;
	class BatchNormNode;
	class EltwiseNode;
	class ReluNode;
	class InnerProductNode;
	class SoftmaxNode;
	class AccuracyNode;

	class ImageDataParam;
	class TransformParam;
	class ConvolutionParam;
	class PoolingParam;
	class BatchNormParam;
	class InnerProductParam;
	class EltwiseParam;
	class AccuracyParam;

	class BiasFiller;

	CREATE_NODE_PARSER_CLASS_2(ImageDataNodeParser, ImageDataNode, ImageDataParam, TransformParam)
	CREATE_NODE_PARSER_CLASS_1(ConvolutionNodeParser, ConvolutionNode, ConvolutionParam)
	CREATE_NODE_PARSER_CLASS_1(PoolingNodeParser, PoolingNode, PoolingParam)
	CREATE_NODE_PARSER_CLASS_1(BatchNormNodeParser, BatchNormNode, BatchNormParam)
	CREATE_NODE_PARSER_CLASS_0(ReluNodeParser, ReluNode)
	CREATE_NODE_PARSER_CLASS_2(InnerProductNodeParser, InnerProductNode, InnerProductParam, BiasFiller)
	CREATE_NODE_PARSER_CLASS_1(EltwiseNodeParser, EltwiseNode, EltwiseParam)
	CREATE_NODE_PARSER_CLASS_0(SoftmaxNodeParser, SoftmaxNode)
	CREATE_NODE_PARSER_CLASS_1(AccuracyNodeParser, AccuracyNode, AccuracyParam)

	template <> inline
	void Parser::ParseLines<string>(map<string, string*> key_value, string stop_str, bool need_fallback) {
		string line = "";
		stringstream word;
		string key = "";
		string value = "";
		streampos pos = 0;
		map<string, bool> parsed_map;

		while (true) {
			pos = m_pfile->TellPos();
			line = m_pfile->ReadLine();
			if (line == "") {
				break;
			}
			word = stringstream(line);
			word >> key;
			if (key == stop_str || key_value.count(key) <= 0) {
				break;
			}
			word >> value;
			value = common::strip(value, "\"");
			*key_value[key] = value;
			parsed_map[key] = true;
			if (parsed_map.size() == key_value.size())
				return;
		}
		if (need_fallback) {
			m_pfile->SeekPos(pos);
		}
	};

	template <> inline
	void Parser::ParseLinesV<string>(map<string, vector<string>*> key_value, string stop_str, bool need_fallback) {
		string line = "";
		stringstream word;
		string key = "";
		string value = "";
		streampos pos = 0;

		while (true) {
			pos = m_pfile->TellPos();
			line = m_pfile->ReadLine();
			if (line == "") {
				break;
			}
			word = stringstream(line);
			word >> key;
			if (key == stop_str || key_value.count(key) <= 0) {
				break;
			}
			word >> value;
			value = common::strip(value, "\"");
			(*key_value[key]).push_back(value);
		}
		if (need_fallback) {
			m_pfile->SeekPos(pos);
		}
	};

}

#endif // !__PARSER_H__