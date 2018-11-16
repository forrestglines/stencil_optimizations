#ifndef TYPE_NAME_H_
#define TYPE_NAME_H_

template <class T>
inline const char* TypeName(){
	return "TYPE_NOT_ENABLED";
}

#define ENABLE_TYPENAME(A) template<> inline const char* TypeName<A>() { return #A; };
ENABLE_TYPENAME(int);
ENABLE_TYPENAME(unsigned int);
ENABLE_TYPENAME(long);
ENABLE_TYPENAME(float);
ENABLE_TYPENAME(double);



#endif
