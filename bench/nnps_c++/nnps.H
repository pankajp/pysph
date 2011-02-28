// ==================================================================
// $Id$
//
// Defines the basic Bin used to index points
// ==================================================================

#ifndef NNPS_H
#define NNPS_H 1

#include <iostream>

#include <vector>
#include <map>

#include <math.h>
#include <assert.h>

namespace nnps {

  struct Point

  {
    //public
    
    int x, y, z;
    
  };

  bool operator < (Point, Point);

  /*!

    \class Bin
    \brief A simple data structure to store particle indices.
    \date $Date$
    \version $Revision$

  */

// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  class Bin {

  public:

    /*! Constructor.
      \param id specifies an index for the cell

    */

    Bin (Point point);

    Bin() {}

    ~Bin() {}

    /* print information */
    void print_stats(void);

    /* The vector holding the particle indices */
    std::vector<std::size_t> indices;

    /* The expected size for the indices */
    int np;

    /* The Point index */
    Point point;

  };
// >>>>>>>>>>>>>>> End of class Bin >>>>>>>>>>>>>>>>

  /*!

    \class BinManager
    \brief Data structure to hold all the bins
    \date $Date$
    \version $Revision$

  */
  // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  class BinManager {

  public:

    /* the map holding the bins */
    std::map<Point, Bin*> bins;

    /* The data arrays */
    std::vector<double> *x, *y;

    /* The length of the arrays */
    std::size_t np;

    /* cell sizes */
    double sx, sy;

    /* gloabl bounds */
    double Mx, mx, My, my;

    /* gloabl bounds for cell indices*/
    long Mcx, mcx, Mcy, mcy;
      
    /* number of cells */
    std::size_t ncx, ncy;

    /* a list of bin ids */
    std::map< std::size_t, std::vector<std::size_t> > bin_list;

    /*! Constructor.
      \param x specifies the x coordinate positions
      \param y specifies the y coordinate positions
      \param sx specifies the cell size in x direction
      \param sy specifies the cell size in y direction

    */

    BinManager(std::vector<double>* x, 
	       std::vector<double>* y,
	       double sx, double sy);

    BinManager() {}
    
    ~BinManager() {
      this->clear();
    }

    void clear()
    {
      std::map<Point, nnps::Bin*>::iterator iter;
      for (iter = this->bins.begin(); iter != this->bins.end(); iter++)
	{
	  delete iter->second;
	}

      this->bins.clear();
    }
    
    /*! bin particles  */
    
    void bin_particles();

    /*! find_bounds */
    
    void find_bounds();

    /*! find_num_cells */
    
    void find_num_cells();

    /* Get the linear bin id from the cell id*/
    
    inline std::size_t cid2bid (const long ix, const long iy)
    {
      return (iy-this->mcy) * this->ncx + (ix-this->mcx);
    }

    inline void bid2cid (const std::size_t bid, long* ix, long* iy)
    {
      (*ix) = (bid % this->ncx) + this->mcx;
      (*iy) = (bid - (*ix) + this->mcx)/this->ncx + this->mcy;
    }

    /* Print information about the bin manager */    
    void get_stats(long*, long*);

  };
  // >>>>>>>>>>>>>>> End of class BinManager >>>>>>>>>>>>>>>>

  /*!

    \class NNPS
    \brief Nearest Neighbor Particle Search
    \date $Date$
    \version $Revision$

  */

  class NNPS {

  public:

    /* the bin manager */
    BinManager* bin_manager;

    /*  a vector for the adjacent cells */
    std::vector<Point> cell_list;

    /* a vector for the nearest particles  */
    std::vector<std::size_t> neighbors;

    /*! Constructor.
      \param bin_manager specifies the bin manager used for binning.

    */

    NNPS(BinManager* bin_manager);
    
    NNPS() {}
    
    ~NNPS() {}
    
    /* function to get nearest particles */
    void get_nearest_particles(const double xp, const double yp, 
			       const double radius);
    
    /* get adjacent cells */
    void get_adjacent_cells(const double xp, const double yp);

  };
  // >>>>>>>>>>>>>>> End of class NNPS >>>>>>>>>>>>>>>>
  
} //namespace nnps

#endif // NNPS_H
