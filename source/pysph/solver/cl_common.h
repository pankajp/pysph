// common function definitions for OpenCL 

#ifndef CL_COMMON
#define CL_COMMON

typedef struct {
  int sizes[3];
} Size;

unsigned int prod(Size* size)
{
  return size->sizes[0] * size->sizes[1] * size->sizes[2];
}

void get_group_information(unsigned int work_dim, Size* local_size,
			   Size* num_groups, Size* local_id, Size* group_id)
{
  for (unsigned int i = 0; i < work_dim; ++i)
    {
      local_size->sizes[i] = get_local_size(i);
      num_groups->sizes[i] = get_num_groups(i);
      local_id->sizes[i] = get_local_id(i);
      group_id->sizes[i] = get_group_id(i);

    } // for i
}

unsigned int get_gid(Size* group_id, Size* num_groups, 
		     Size* local_id, Size* local_size,
		     unsigned int group_size)
{
  unsigned int gid;

  gid = group_id->sizes[0] * group_size +
    group_id->sizes[1] * num_groups->sizes[0]*group_size +
    group_id->sizes[2] * num_groups->sizes[0]*num_groups->sizes[1]*group_size;
  
  gid += local_id->sizes[0] +
    local_id->sizes[1] * local_size->sizes[0] +
    local_id->sizes[2] * local_size->sizes[0]*local_size->sizes[1];

  return gid;
}

#endif CL_COMMON
