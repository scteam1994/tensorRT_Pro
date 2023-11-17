/*
 这个文件在windows下有效，linux下请忽略
*/

#if defined(_WIN32)
#	define U_OS_WINDOWS
#else
#   define U_OS_LINUX
#endif

#ifdef U_OS_WINDOWS
#if defined(_DEBUG)
#	pragma comment(lib, "opencv_world346d.lib")
#else
#	pragma comment(lib, "opencv_world346.lib")
#endif

#endif // U_OS_WINDOWS