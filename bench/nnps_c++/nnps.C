// ===================================================================
// $Id$
// ===================================================================

#include "nnps.H"

#define BIG 10000
#define SMALL -BIG

namespace nnps {
  
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

  bool operator < (Voxel p1, Voxel p2)
  {
    if (p1.x < p2.x)
      return true;

    else if (p1.x > p2.x)
      return false;
    
    else
      {
	if (p1.y < p2.y)
	  return true;
	
	else if (p1.y > p2.y)
	  return false;
	
	else
	  {
	    if (p1.z < p2.z)
	      return true;
	    
	    else if (p1.z > p2.z)
	      return false;

	    else
	      return false;
	  }
      }
  }

  Bin:: Bin(Voxel voxel){
    this->voxel = voxel;
  }  
  
  void Bin:: print_stats(void)
  {
    std::cout << "Bin (" << this->voxel.x << ", " 
	      << this->voxel.y << ", " << this->voxel.z << ")";
    
    std::cout << "\t Number of particles: " << this->indices.size() <<std::endl;
  }    

  std::size_t Bin:: get_size(void)
  {
    return this->indices.size();
  }
    
  // >>>>>>>>>>>>>> End of Bin definitions >>>>>>>>>>>>>

  BinManager:: BinManager(std::vector<double>* x, 
			  std::vector<double>* y,
			  double sx, double sy){
    
    this->x = x;
    this->y = y;

    assert(this->x->size() == this->y->size());

    this->np = this->x->size();
    
    this->sx = sx;
    this->sy = sy;

    this->find_bounds();

    this->find_num_cells();

  }

  void BinManager:: find_bounds(){
    
    this->Mx = SMALL;
    this->mx = BIG;

    this->My = SMALL;
    this->my = BIG;

    for (std::size_t i = 0; i < this->np; ++i){

      if (x->at(i) > this->Mx){
	this->Mx = x->at(i);}

      if (x->at(i) < this->mx){
	this->mx = x->at(i);}
      
      if (y->at(i) > this->My){
	this->My = y->at(i);}

      if (y->at(i)< this->my){
	this->my = y->at(i);}
    }
  }

  void BinManager:: bin_particles(){

    long ix, iy, iz;

    this->clear();

    std::map<Voxel, nnps::Bin*>::const_iterator iter;

    for ( std::size_t i = 0; i < this->np; i++ )
      {
	ix = floor(this->x->at(i)/this->sx);
	iy = floor(this->y->at(i)/this->sx);
	iz = 0;
      
	Voxel voxel = {ix, iy, iz};

	iter = this->bins.find(voxel);

	if (iter == this->bins.end())
	  {
	    Bin *bin = new Bin(voxel);
	    bin->indices.push_back(i);
	    this->bins[voxel] = bin;
	  }
	else
	    (*iter).second->indices.push_back(i);
      }
  }
  
  void BinManager:: find_num_cells(){

    this->ncx = 1;
    this->ncy = 1;

    double Mx = this->Mx;
    double mx = this->mx;
    double My = this->My;
    double my = this->my;

    double sx = this->sx;
    double sy = this->sy;

    this->Mcx = floor(Mx/sx);
    this->mcx = floor(mx/sx);

    this->Mcy = floor(My/sy);
    this->mcy = floor(my/sy);

    this->ncx = 1 + (this->Mcx - this->mcx);
    this->ncy = 1 + (this->Mcy - this->mcy);

  }

  void BinManager:: get_stats(long* n_bins, long* np_max){
    std::map<Voxel, Bin*>::iterator iter;

    size_t nbins = 0;
    size_t max_np = 0;
    
    std::vector<std::size_t> indices;
        
    for ( iter = this->bins.begin(); iter != this->bins.end(); ++iter)
      {
	indices = iter->second->indices;
	if (indices.size() > max_np)
	  max_np = indices.size();
	
	nbins ++;
      }
    
    (*np_max) = max_np;
    (*n_bins) = nbins;
  }

  void BinManager:: get_sizes(double* sx, double* sy)
  {
    (*sx) = this->sx;
    (*sy) = this->sy;
  }

  void BinManager:: get_position(double* xp, double* yp, std::size_t i)
  {
    
    (*xp) = this->x->at(i);
    (*yp) = this->y->at(i);
  }

// >>>>>>>>>>>>>> End of BinManager definitions >>>>>>>>>>>>>

  NNPS:: NNPS(BinManager* bin_manager)
  {
    this->bin_manager = bin_manager;
    this->cell_list.resize(9);
  }

  void NNPS:: get_adjacent_cells(const double xp, const double yp){

    int count;
    long ix, iy, i, j;
    double sx = this->bin_manager->sx;
    double sy = this->bin_manager->sy;

    ix = floor(xp/sx);
    iy = floor(yp/sy);

    count = 0;

    for (i = ix-1; i <= ix + 1; i++)
      {
	for (j = iy-1; j <= iy + 1; j++)
	  {
	    Voxel voxel = {i, j, 0};
	    this->cell_list[count] = voxel;

	    count++;
	    
	  }
      }     
  }

  void NNPS:: get_nearest_particles(const double xp, const double yp, 
				    const double radius, 
				    std::vector<std::size_t> *nbrs){

    this->neighbors.clear();
    this->get_adjacent_cells(xp, yp);
    
    std::map<Voxel, nnps::Bin*>::iterator iter;
    std::vector<std::size_t> indices;

    std::vector<double> *x = this->bin_manager->x;
    std::vector<double> *y = this->bin_manager->y;

    long index;
    double xj, yj, dist;

    Voxel voxel;

    nbrs->clear();

    for ( std::size_t i = 0; i < this->cell_list.size(); ++i )
      {
	voxel = this->cell_list[i];

	iter = this->bin_manager->bins.find(voxel);
	
	if (iter == this->bin_manager->bins.end())
	  continue;
	
	indices = iter->second->indices;

	for (std::size_t j = 0; j < indices.size(); j++)
	  {
	    index = indices[j];
	    
	    xj = (*x)[index];
	    yj = (*y)[index];
	    
	    dist = (xp-xj)*(xp-xj) + (yp-yj)*(yp-yj);

	    if (dist < radius*radius)
	      {
		//this->neighbors.push_back(index);
		nbrs->push_back(index);
		
	      }
	  }
      }

  }


} //namespace nnps
