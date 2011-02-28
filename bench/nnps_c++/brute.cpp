// Brute force neighbor search
// // g++ nnps.C brute.cpp -O3 -o brute
// ./brute <np>

#include "nnps.H"
#include<cstdlib>

using namespace std;

int main(int argc, char* argv[]){
  
  long np = 1000000;
  if (argc > 1)
      np = atol(argv[1]);

  cout << "Brute force neighbor search for " 
       << np/1e6 << " million particles " << endl;
  
  vector<double> x(np, 0.0);
  vector<double> y(np, 0.0);

  srand(time(0));

  for (long i = 0; i < np; ++i)
    {
      int sign1 = (rand()/(float)RAND_MAX > rand()/(float)RAND_MAX) ? 1 : -1;
      int sign2 = (rand()/(float)RAND_MAX > rand()/(float)RAND_MAX) ? 1 : -1;

      x[i] = sign1 * rand() /(float)RAND_MAX;
      y[i] = sign2 * rand() /(float)RAND_MAX;

    }

  // h ~ 2*vol_per_particle
  // rad = 2*h => rad ~ 4*vol_per_particle

  int dim = 2;
  double volume = 4.0;
  double nd = pow(volume/(float)np, 1.0/dim);
  double rad = 6*nd;

  double radius = rad;

  double xi, yi, xj, yj;
  double dist;

  for (size_t i = 0; i < x.size(); i++)
    {
      xi = x[i];
      yi = y[i];

      vector<size_t> nbrs(0);
      
      for (size_t j = 0; j < x.size(); j++)
	{	  
	  xj = x[j];
	  yj = y[j];
	  
	  dist = (xi-xj)*(xi-xj) + (yi-yj)*(yi-yj);
	  
	  if (dist < radius*radius);
	  nbrs.push_back(j);
	}
    }

  return 0;

}



