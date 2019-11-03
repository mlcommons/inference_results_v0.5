#ifndef CREATE_SIMPLE_ATTR_SET_GET
#define CREATE_SIMPLE_ATTR_SET_GET(attr_name, attr_type) \
protected: \
attr_type  attr_name; \
public: \
attr_type& get_##attr_name() { return attr_name; }; \
void set_##attr_name(attr_type value) { attr_name = move(value); };
#endif

#ifndef CREATE_SIMPLE_ATTR_GET
#define CREATE_SIMPLE_ATTR_GET(attr_name, attr_type) \
protected: \
attr_type  attr_name; \
public: \
attr_type& get_##attr_name() { return attr_name; };
#endif

#ifndef CREATE_SIMPLE_CLASS_GET_1
#define CREATE_SIMPLE_CLASS_GET_1(class_name, param_name1, param_type1) \
class class_name { \
	CREATE_SIMPLE_ATTR_GET(m_##param_name1, param_type1); \
                                                          \
public: \
	class_name(param_type1 param_name1) : m_##param_name1(param_name1) {} \
	virtual ~class_name() {}; \
};
#endif

#ifndef CREATE_SIMPLE_CLASS_GET_2
#define CREATE_SIMPLE_CLASS_GET_2(class_name, param_name1, param_type1, param_name2, param_type2) \
class class_name { \
	CREATE_SIMPLE_ATTR_GET(m_##param_name1, param_type1); \
	CREATE_SIMPLE_ATTR_GET(m_##param_name2, param_type2); \
                                                          \
public: \
	class_name(param_type1 param_name1, param_type2 param_name2) \
		: m_##param_name1(param_name1), m_##param_name2(param_name2) {} \
	virtual ~class_name() {}; \
};
#endif

#ifndef CREATE_SIMPLE_CLASS_GET_3
#define CREATE_SIMPLE_CLASS_GET_3(class_name, param_name1, param_type1, param_name2, param_type2, param_name3, param_type3) \
class class_name { \
	CREATE_SIMPLE_ATTR_GET(m_##param_name1, param_type1); \
	CREATE_SIMPLE_ATTR_GET(m_##param_name2, param_type2); \
	CREATE_SIMPLE_ATTR_GET(m_##param_name3, param_type3); \
                                                          \
public: \
	class_name(param_type1 param_name1, param_type2 param_name2, param_type3 param_name3) \
		: m_##param_name1(param_name1), m_##param_name2(param_name2), m_##param_name3(param_name3) {} \
	virtual ~class_name() {}; \
};
#endif

#ifndef CREATE_SIMPLE_CLASS_GET_6
#define CREATE_SIMPLE_CLASS_GET_6(class_name, param_name1, param_type1, param_name2, param_type2, param_name3, param_type3, \
	param_name4, param_type4, param_name5, param_type5, param_name6, param_type6) \
class class_name { \
	CREATE_SIMPLE_ATTR_GET(m_##param_name1, param_type1); \
	CREATE_SIMPLE_ATTR_GET(m_##param_name2, param_type2); \
	CREATE_SIMPLE_ATTR_GET(m_##param_name3, param_type3); \
	CREATE_SIMPLE_ATTR_GET(m_##param_name4, param_type4); \
	CREATE_SIMPLE_ATTR_GET(m_##param_name5, param_type5); \
	CREATE_SIMPLE_ATTR_GET(m_##param_name6, param_type6); \
                                                          \
public: \
	class_name(param_type1 param_name1, param_type2 param_name2, param_type3 param_name3, param_type4 param_name4, param_type5 param_name5, param_type6 param_name6) \
		: m_##param_name1(param_name1), m_##param_name2(param_name2), m_##param_name3(param_name3), \
		  m_##param_name4(param_name4), m_##param_name5(param_name5), m_##param_name6(param_name6){} \
	virtual ~class_name() {}; \
};
#endif

#ifndef CREATE_NODE_CLASS_0
#define CREATE_NODE_CLASS_0(class_name) \
class class_name : public Node { \
public: \
	class_name(string name, string type, vector<string> bottom, vector<string> top) \
		: Node(name, type, bottom, top) { \
	}; \
	~class_name() {}; \
};
#endif // !CREATE_NODE_CLASS_0

#ifndef CREATE_NODE_CLASS_1
#define CREATE_NODE_CLASS_1(class_name, param_name1, param_type1) \
class class_name : public Node { \
	CREATE_SIMPLE_ATTR_GET(m_##param_name1, param_type1); \
\
public: \
	class_name(string name, string type, vector<string> bottom, vector<string> top, param_type1 param_name1) \
		: Node(name, type, bottom, top), m_##param_name1(param_name1) { \
	}; \
	~class_name() {}; \
};
#endif // !CREATE_NODE_CLASS_1

#ifndef CREATE_NODE_CLASS_2
#define CREATE_NODE_CLASS_2(class_name, param_name1, param_type1, param_name2, param_type2) \
class class_name : public Node { \
	CREATE_SIMPLE_ATTR_GET(m_##param_name1, param_type1); \
	CREATE_SIMPLE_ATTR_GET(m_##param_name2, param_type2); \
\
public: \
	class_name(string name, string type, vector<string> bottom, vector<string> top, \
		param_type1 param_name1, param_type2 param_name2) \
		: Node(name, type, bottom, top), m_##param_name1(param_name1), m_##param_name2(param_name2) { \
	}; \
	~class_name() {}; \
};
#endif // !CREATE_NODE_CLASS_2

#ifndef CREATE_NODE_CLASS_3
#define CREATE_NODE_CLASS_3(class_name, param_name1, param_type1, \
	param_name2, param_type2, param_name3, param_type3) \
class class_name : public Node { \
	CREATE_SIMPLE_ATTR_GET(m_##param_name1, param_type1); \
	CREATE_SIMPLE_ATTR_GET(m_##param_name2, param_type2); \
	CREATE_SIMPLE_ATTR_GET(m_##param_name3, param_type3); \
\
public: \
	class_name(string name, string type, vector<string> bottom, vector<string> top, \
		param_type1 param_name1, param_type2 param_name2, param_type3 param_name3) \
		: Node(name, type, bottom, top), m_##param_name1(param_name1), \
		m_##param_name2(param_name2), m_##param_name3(param_name3) { \
	}; \
	~class_name() {}; \
};
#endif // !CREATE_NODE_CLASS_3

#ifndef CREATE_NODE_PARSER_CLASS_0
#define CREATE_NODE_PARSER_CLASS_0(class_name, node_class) \
class class_name : public NodeParser { \
public: \
	class_name(InputFile* pfile, Config* pconf); \
	~class_name(); \
\
	node_class Run(Node node); \
};
#endif // !CREATE_NODE_PARSER_CLASS_1

#ifndef CREATE_NODE_PARSER_CLASS_1
#define CREATE_NODE_PARSER_CLASS_1(class_name, node_class, param_class1) \
class class_name : public NodeParser { \
public: \
	class_name(InputFile* pfile, Config* pconf); \
	~class_name(); \
\
	node_class Run(Node node); \
	param_class1 Parse##param_class1(); \
};
#endif // !CREATE_NODE_PARSER_CLASS_1

#ifndef CREATE_NODE_PARSER_CLASS_2
#define CREATE_NODE_PARSER_CLASS_2(class_name, node_class, param_class1, param_class2) \
class class_name : public NodeParser { \
public: \
	class_name(InputFile* pfile, Config* pconf); \
	~class_name(); \
\
	node_class Run(Node node); \
	param_class1 Parse##param_class1(); \
	param_class2 Parse##param_class2(); \
};
#endif // !CREATE_NODE_PARSER_CLASS_2
