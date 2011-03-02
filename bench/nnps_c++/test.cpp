// Test the correctness of the neighbor search algorithm
// Compile it as g++ nnps.C test.cpp -O3 -o test
// to run: ./test <np>

#include "nnps.H"
#include<cstdlib>
#include <algorithm>

using namespace std;

using namespace std;

vector<double> *xp, *yp;

nnps::NNPS *nps;
nnps::BinManager* bin_manager;

double radius;
size_t id;


void verify_neighbors(void){

  double x = (*xp)[id];
  double y = (*yp)[id];
  
  double dist = 0.0;
  vector<size_t> nbrs(0);

  // Get neighbors by brute force

  for (size_t i = 0; i < xp->size(); i++)
    {
      dist = (x - xp->at(i))*(x - xp->at(i)) + (y - yp->at(i))*(y - yp->at(i));

      if (dist < radius*radius)
	{
	  nbrs.push_back(i);
	}
    }
  
  // Get neighbors using NNPS

  vector<size_t> inbrs(0);

  nps->get_nearest_particles(x, y, radius, &inbrs);

  //vector<size_t> nnps_nbrs = nps->neighbors;

  //assert (nnps_nbrs.size() == nbrs.size());

  assert (inbrs.size() == nbrs.size());

  //sort( nnps_nbrs.begin(), nnps_nbrs.end() );

  sort( inbrs.begin(), inbrs.end() );  

  for (size_t k = 0; k < nbrs.size(); k++)
    assert (nbrs[k] == inbrs[k]);
}

int main(int argc, char* argv[]){
   long np = 10000;
  if (argc > 1)
      np = atol(argv[1]);

  cout << np/1e6 << " million particles " << endl;
  
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

  xp = &x;
  yp = &y;

  // h ~ 2*vol_per_particle
  // rad = 2*h

  int dim = 2;
  double volume = 4.0;
  double nd = pow(volume/(float)np, 1.0/dim);
  double rad = 6*nd;

  double sx = rad;
  double sy = rad;
  radius = rad;

  bin_manager = new nnps::BinManager(&x,&y,sx,sy);
  bin_manager->bin_particles();

  nps = new nnps::NNPS(bin_manager);
  
  for (id = 0; id < x.size(); id++)
    verify_neighbors();

  cout << "OK" << endl;

  delete bin_manager;
  delete nps;
  
  return 0;

}


