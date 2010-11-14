#include "cPoint.h"
#include <cmath>
int IntPoint_maxint = std::pow(2,20);

long hash(cIntPoint p)
{
    long ret = p.x + IntPoint_maxint;
    ret = 2 * IntPoint_maxint * ret + p.y + IntPoint_maxint;
    return 2 * IntPoint_maxint * ret + p.z + IntPoint_maxint;
    return ret;
};

// This is needed for cIntPoint to be usable in c++ stl map
bool operator<(cIntPoint pa, cIntPoint pb)
{
  return hash(pa) < hash(pb);
};
