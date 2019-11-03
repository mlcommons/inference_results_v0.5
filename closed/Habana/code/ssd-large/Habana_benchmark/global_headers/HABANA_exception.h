#ifndef _HABANA_EXCEPTIONS_
#define _HABANA_EXCEPTIONS_
#include <string>

enum class HABANA_benchStatus
{
        HABANA_SUCCESS,
        HABANA_FAIL,
        HABANA_FAIL_DUE_TO_INPUT_FILE_PROBLEM,
        HABANA_FAIL_DUE_TO_INPUT_FILE_SIZE,
        HABANA_FAIL_DUE_VECOTR_SIZE_ERROR
};


class HABANA_benchException:public std::exception
{
    public:
            HABANA_benchException(const std::string m):m_msg(m){}
            ~HABANA_benchException(void) = default;
            const char* what(){return m_msg.c_str();}
    private:
            std::string m_msg;
};
#endif