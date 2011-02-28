// search for nearest neighbors
// g++ nnps.C nnps.cpp -O3 -o nnps
// ./nnps <np>

#include "nnps.H"
#include<cstdlib>

using namespace std;

vector<double> *xp, *yp;

nnps::NNPS *nps;
nnps::BinManager* bin_manager;

double radius;

void nnps_neighbor_search(void)
{  
  double xi, yi;

  for (size_t id = 0; id < xp->size(); id++)
    {
      xi = xp->at(id);
      yi = yp->at(id);

      nps->get_nearest_particles(xi, yi, radius);
    }
}

void bin_particles(void)
{
  bin_manager->bin_particles();
}

int main(int argc, char* argv[]){
  
  long np = 1000000;
  if (argc > 1)
      np = atol(argv[1]);

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

  double sx = 0.4/2;
  double sy = 0.4/2;
  radius = sx;

  bin_manager = new nnps::BinManager(&x,&y,sx,sy);

  bin_particles();

  nps = new nnps::NNPS(bin_manager);
  
  nnps_neighbor_search();

  long np_max, nbins;
  double avg;

  bin_manager->get_stats(&nbins, &np_max);

  avg = (float)x.size()/(float)nbins ;
  
  cout << "Np: " << np << " ( " << (float)np/1e6 << " million) " 
       << "  Number of bins: " << nbins << "  Particles/bin (avg): " << avg
       << endl << "Max bin size: " << np_max << endl; 

  delete bin_manager;
  delete nps;

  return 0;

}
