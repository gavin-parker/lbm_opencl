/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(jj)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(ii) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   d2q9-bgk.exe input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<string.h>
#ifdef __unix__
#include<sys/time.h>
#include<sys/resource.h>
#else
#include <time.h>
#endif
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"
#define OCLFILE         "kernels.cl"
#define INDEX(ii,jj,nx,ny,speed) (((nx)*(ny)*(speed))+((ii)*(nx)+(jj)))

#ifndef BLOCK_I
	#define BLOCK_I 16
#endif
#ifndef BLOCK_J
	#define BLOCK_J 16
#endif
/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
} t_param;

/* struct to hold OpenCL objects */
typedef struct
{
  cl_device_id      device;
  cl_context        context;
  cl_command_queue  queue;

  cl_program program;
  cl_kernel  accelerate_flow;
  cl_kernel  propagate;
  cl_kernel  collision;
  cl_kernel  rebound;
  cl_kernel  av_velocity;


  cl_mem cells;
  cl_mem tmp_cells;
  cl_mem obstacles;
  cl_mem total_vel;
  int workGroups;
  size_t workGroupSize;

} t_ocl;


/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, float** cells_ptr, float** tmp_cells_ptr,
               short** obstacles_ptr, float** av_vels_ptr, t_ocl* ocl);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
int timestep(const t_param params, float* cells, float* tmp_cells, short* obstacles, t_ocl ocl, int flip, int tt);
int accelerate_flow(const t_param params, float* cells, short* obstacles, t_ocl ocl, int flip);
int rebound(const t_param params, float* cells, float* tmp_cells, short* obstacles, t_ocl ocl, int flip);
int collision(const t_param params, short* obstacles, t_ocl ocl, int flip, int tt);
int write_values(const t_param params, float* cells, short* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, float** cells_ptr, float** tmp_cells_ptr,
	short** obstacles_ptr, float** av_vels_ptr, t_ocl ocl);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, float* cells);

/* compute average velocity */
int av_velocity(const t_param params, float* cells, short* obstacles, t_ocl ocl, float* av_vels);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, float* cells, short* obstacles, t_ocl ocl, float vel);

/* utility functions */
void checkError(cl_int err, const char *op, const int line);
void die(const char* message, const int line, const char* file);
void usage(const char* exe);
void set_kernel_args(const t_param params, t_ocl ocl);
cl_device_id selectOpenCLDevice();

int total_cells;
float* total_vel = NULL;
int total_obstacles = 0;
char* options = " -cl-mad-enable -cl-unsafe-math-optimizations -cl-finite-math-only -cl-fast-relaxed-math";
/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
	char*    paramfile = NULL;    /* name of the input parameter file */
	char*    obstaclefile = NULL; /* name of a the input obstacle file */
	t_param  params;              /* struct to hold parameter values */
	t_ocl    ocl;                 /* struct to hold OpenCL objects */
	float* cells = NULL;    /* grid containing fluid densities */
	float* tmp_cells = NULL;    /* scratch space */
	short*     obstacles = NULL;    /* grid indicating which cells are blocked */
	float* av_vels = NULL;     /* a record of the av. velocity computed for each timestep */
	cl_int err;
#ifdef __unix__
	struct timeval timstr;        /* structure to hold elapsed time */
	struct rusage ru;             /* structure to hold CPU time--system and user */

	double tic, toc;              /* floating point numbers to calculate elapsed wallclock time */
	double usrtim;                /* floating point number to record elapsed user CPU time */
	double systim;                /* floating point number to record elapsed system CPU time */
#endif
	/* parse the command line */
	if (argc != 3)
	{
		usage(argv[0]);
	}
	else
	{
		paramfile = argv[1];
		obstaclefile = argv[2];
	}

	/* initialise our data structures and load values from file */
	initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels, &ocl);
	set_kernel_args(params, ocl);

	/* iterate for maxIters timesteps */
#ifdef __unix__
	gettimeofday(&timstr, NULL);
	tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
#else
	clock_t start = clock();
#endif

	// Write cells to OpenCL buffer
	err = clEnqueueWriteBuffer(
		ocl.queue, ocl.cells, CL_FALSE, 0,
		sizeof(float) * params.nx * params.ny*NSPEEDS, cells, 0, NULL, NULL);
	checkError(err, "writing cells data", __LINE__);

	// Write obstacles to OpenCL buffer00
	err = clEnqueueWriteBuffer(
		ocl.queue, ocl.obstacles, CL_TRUE, 0,
		sizeof(cl_short) * params.nx * params.ny, obstacles, 0, NULL, NULL);
	checkError(err, "writing obstacles data", __LINE__);


	int flip = 0;
	for (int tt = 0; tt < params.maxIters; tt++)
	{
		timestep(params, cells, tmp_cells, obstacles, ocl, flip, tt);
		//av_vels[tt] = av_velocity(params, cells, obstacles, ocl);
		flip = !flip;
		//err = clFinish(ocl.queue);
#ifdef DEBUG
		printf("==timestep: %d==\n", tt);
		printf("av velocity: %.12E\n", av_vels[tt]);
		printf("tot density: %.12E\n", total_density(params, cells));
#endif
	}
	if (!flip){
		err = clEnqueueReadBuffer(
		  ocl.queue, ocl.cells, CL_TRUE, 0,
		  sizeof(float) * params.nx * params.ny*NSPEEDS, cells, 0, NULL, NULL);
	}
	else {
		err = clEnqueueReadBuffer(
			ocl.queue, ocl.tmp_cells, CL_TRUE, 0,
			sizeof(float) * params.nx * params.ny*NSPEEDS, cells, 0, NULL, NULL);
	}

  av_velocity(params, cells, obstacles, ocl, av_vels);

  checkError(err, "reading tmp_cells data", __LINE__);
   

#ifdef  __unix__
  gettimeofday(&timstr, NULL);
  toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  getrusage(RUSAGE_SELF, &ru);
  timstr = ru.ru_utime;
  usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  timstr = ru.ru_stime;
  systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  double time = toc - tic;
#else
  clock_t end = clock();
  float time = (float)(end - start) / CLOCKS_PER_SEC;
#endif
  /* write final values and free memory */
  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles, ocl, av_vels[params.maxIters-1]));
#ifdef __unix__
  printf("Elapsed time:\t\t\t%.6f (s)\n", time);
  printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
  printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
#else
  printf("Elapsed time:\t\t\t%.6f (s)\n", time);

#endif
  write_values(params, cells, obstacles, av_vels);
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels, ocl);

  return EXIT_SUCCESS;
}

int timestep(const t_param params, float* cells, float *tmp_cells, short* obstacles, t_ocl ocl, int flip, int tt)
{
  cl_int err;

  // Write cells to device
  accelerate_flow(params, cells, obstacles, ocl, flip);
  collision(params, obstacles, ocl, flip, tt);
  //err = clFinish(ocl.queue);

  return EXIT_SUCCESS;
}

int accelerate_flow(const t_param params, float* cells, short* obstacles, t_ocl ocl, int flip)
{
  cl_int err;
  //err = clFinish(ocl.queue);
  // Set kernel arguments
  if (!flip) {
	  err = clSetKernelArg(ocl.accelerate_flow, 0, sizeof(cl_mem), &ocl.cells);
  }
  else {
	  err = clSetKernelArg(ocl.accelerate_flow, 0, sizeof(cl_mem), &ocl.tmp_cells);
  }
  checkError(err, "setting accelerate_flow arg 0", __LINE__);


  // Enqueue kernel
  size_t global[1] = {params.nx};
  err = clEnqueueNDRangeKernel(ocl.queue, ocl.accelerate_flow,
                               1, NULL, global, NULL, 0, NULL, NULL);
  checkError(err, "enqueueing accelerate_flow kernel", __LINE__);

  // Wait for kernel to finish
  //err = clFinish(ocl.queue);
  //checkError(err, "waiting for accelerate_flow kernel", __LINE__);

  return EXIT_SUCCESS;
}



int collision(const t_param params, short* obstacles, t_ocl ocl, int flip , int tt)
{
	cl_int err;

	// Set kernel arguments
	err = clSetKernelArg(ocl.collision, flip, sizeof(cl_mem), &ocl.cells);
	checkError(err, "setting collision arg 0", __LINE__);
	err = clSetKernelArg(ocl.collision, !flip, sizeof(cl_mem), &ocl.tmp_cells);
	checkError(err, "setting collision arg 1", __LINE__);
	err = clSetKernelArg(ocl.collision, 7, sizeof(cl_int), &tt);
	checkError(err, "setting collision arg 7", __LINE__);
	// Enqueue kernel
	size_t global[2] = { params.nx, params.ny };
	size_t local[2] = { 64, 2 };
	err = clEnqueueNDRangeKernel(ocl.queue, ocl.collision,
		2, NULL, global, local, 0, NULL, NULL);

	checkError(err, "enqueueing collision kernel", __LINE__);

	// Wait for kernel to finish
	//err = clFinish(ocl.queue);
	//checkError(err, "waiting for collision kernel", __LINE__);

  return EXIT_SUCCESS;
}

void set_kernel_args(const t_param params,t_ocl ocl) {
	cl_int err;
	err = clSetKernelArg(ocl.collision, 2, sizeof(cl_mem), &ocl.obstacles);
	checkError(err, "setting collision arg 2", __LINE__);
	err = clSetKernelArg(ocl.collision, 3, sizeof(cl_int), &params.nx);
	checkError(err, "setting collision arg 3", __LINE__);
	err = clSetKernelArg(ocl.collision, 4, sizeof(cl_int), &params.ny);
	checkError(err, "setting collision arg 4", __LINE__);
	err = clSetKernelArg(ocl.collision, 5, sizeof(cl_float), &params.omega);
	checkError(err, "setting collision arg 5", __LINE__);
	err = clSetKernelArg(ocl.collision, 6, sizeof(cl_mem), &ocl.total_vel);
	checkError(err, "setting collision arg 6", __LINE__);
	err = clSetKernelArg(ocl.accelerate_flow, 1, sizeof(cl_mem), &ocl.obstacles);
	checkError(err, "setting accelerate_flow arg 1", __LINE__);
	err = clSetKernelArg(ocl.accelerate_flow, 2, sizeof(cl_int), &params.nx);
	checkError(err, "setting accelerate_flow arg 2", __LINE__);
	err = clSetKernelArg(ocl.accelerate_flow, 3, sizeof(cl_int), &params.ny);
	checkError(err, "setting accelerate_flow arg 3", __LINE__);
	err = clSetKernelArg(ocl.accelerate_flow, 4, sizeof(cl_float), &params.density);
	checkError(err, "setting accelerate_flow arg 4", __LINE__);
	err = clSetKernelArg(ocl.accelerate_flow, 5, sizeof(cl_float), &params.accel);
	checkError(err, "setting accelerate_flow arg 5", __LINE__);

}

int av_velocity(const t_param params, float* cells, short* obstacles, t_ocl ocl, float* av_vels)
{
	cl_int err;

	err = clEnqueueReadBuffer(
		ocl.queue, ocl.total_vel, CL_TRUE, 0,
		sizeof(cl_float)*(ocl.workGroups)*params.maxIters, total_vel, 0, NULL, NULL);
	checkError(err, "reading total_vel data", __LINE__);

	int groups = ocl.workGroups;
	for (int i = 0; i < params.maxIters; i++) {
		double tot = 0;
	#pragma simd
		for (int j = 0; j < groups; j++) {
			tot += total_vel[i*groups + j];
		}
		av_vels[i] = tot / (double)total_cells;
	}
	return EXIT_SUCCESS;

}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, float** cells_ptr, float** tmp_cells_ptr,
	short** obstacles_ptr, float** av_vels_ptr, t_ocl *ocl)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */
  char*  ocl_src;        /* OpenCL kernel source */
  long   ocl_size;       /* size of OpenCL kernel source */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  double a;

  retval = fscanf(fp, "%lf\n", &(a));
  params->density = (float) a;

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%lf\n", &(a));
  params->accel = (float) a;

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%lf\n", &(a));
  params->omega = (float) a;

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* main grid */
  *cells_ptr = (float*)malloc(sizeof(float) * (params->ny * params->nx)*NSPEEDS);

  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  *tmp_cells_ptr = (float*)malloc(sizeof(float) * (params->ny * params->nx)*NSPEEDS);

  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(short) * (params->ny * params->nx));

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.0 / 9.0;
  float w1 = params->density      / 9.0;
  float w2 = params->density      / 36.0;

  total_cells = params->ny * params->nx;
  for (int ii = 0; ii < params->ny; ii++)
  {
    for (int jj = 0; jj < params->nx; jj++)
    {
      /* centre */
      (*cells_ptr)[INDEX(ii,jj,params->nx, params->ny,0)] = w0;
      /* axis directions */
      (*cells_ptr)[INDEX(ii, jj, params->nx, params->ny, 1)] = w1;
	  (*cells_ptr)[INDEX(ii, jj, params->nx, params->ny, 2)] = w1;
	  (*cells_ptr)[INDEX(ii, jj, params->nx, params->ny, 3)] = w1;
	  (*cells_ptr)[INDEX(ii, jj, params->nx, params->ny, 4)] = w1;
	  /* diagonals */
	  (*cells_ptr)[INDEX(ii, jj, params->nx, params->ny, 5)] = w2;
	  (*cells_ptr)[INDEX(ii, jj, params->nx, params->ny, 6)] = w2;
	  (*cells_ptr)[INDEX(ii, jj, params->nx, params->ny, 7)] = w2;
	  (*cells_ptr)[INDEX(ii, jj, params->nx, params->ny, 8)] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  for (int ii = 0; ii < params->ny; ii++)
  {
    for (int jj = 0; jj < params->nx; jj++)
    {
      (*obstacles_ptr)[ii * params->nx + jj] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    (*obstacles_ptr)[yy * params->nx + xx] = blocked;
	if (blocked) {
		total_cells--;
		total_obstacles++;
	}
  }
  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);


  cl_int err;

  ocl->device = selectOpenCLDevice();

  // Create OpenCL context
  ocl->context = clCreateContext(NULL, 1, &ocl->device, NULL, NULL, &err);
  checkError(err, "creating context", __LINE__);

  fp = fopen(OCLFILE, "r");
  if (fp == NULL)
  {
    sprintf(message, "could not open OpenCL kernel file: %s", OCLFILE);
    die(message, __LINE__, __FILE__);
  }

  // Create OpenCL command queue
  ocl->queue = clCreateCommandQueue(ocl->context, ocl->device, 0, &err);
  checkError(err, "creating command queue", __LINE__);

  // Load OpenCL kernel source
  fseek(fp, 0, SEEK_END);
  ocl_size = ftell(fp) + 1;
  ocl_src = (char*)malloc(ocl_size);
  memset(ocl_src, 0, ocl_size);
  fseek(fp, 0, SEEK_SET);
  fread(ocl_src, 1, ocl_size, fp);
  fclose(fp);

  // Create OpenCL program
  ocl->program = clCreateProgramWithSource(
    ocl->context, 1, (const char**)&ocl_src, NULL, &err);
  free(ocl_src);
  checkError(err, "creating program", __LINE__);

  // Build OpenCL program
  err = clBuildProgram(ocl->program, 1, &ocl->device, options, NULL, NULL);
  if (err == CL_BUILD_PROGRAM_FAILURE)
  {
    size_t sz;
    clGetProgramBuildInfo(
      ocl->program, ocl->device,
      CL_PROGRAM_BUILD_LOG, 0, NULL, &sz);
    char *buildlog = malloc(sz);
    clGetProgramBuildInfo(
      ocl->program, ocl->device,
      CL_PROGRAM_BUILD_LOG, sz, buildlog, NULL);
    fprintf(stderr, "\nOpenCL build log:\n\n%s\n", buildlog);
    free(buildlog);
  }
  checkError(err, "building program", __LINE__);

  // Create OpenCL kernels
  ocl->accelerate_flow = clCreateKernel(ocl->program, "accelerate_flow", &err);
  checkError(err, "creating accelerate_flow kernel", __LINE__);
  ocl->collision = clCreateKernel(ocl->program, "collision", &err);
  checkError(err, "creating collision kernel", __LINE__);

  // Allocate OpenCL buffers
  ocl->cells = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(float) * params->nx * params->ny*NSPEEDS, NULL, &err);
  checkError(err, "creating cells buffer", __LINE__);
  ocl->tmp_cells = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(float) * params->nx * params->ny*NSPEEDS, NULL, &err);
  checkError(err, "creating tmp_cells buffer", __LINE__);
  ocl->obstacles = clCreateBuffer(
	  ocl->context, CL_MEM_READ_ONLY,
	  sizeof(short) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating tmp_cells buffer", __LINE__);

 
  ocl->workGroupSize = 64*2;
  ocl->workGroups = (params->nx*params->ny) / ocl->workGroupSize;

  printf("workgroup size: %d \n", (int)ocl->workGroupSize);
  printf("workgroup count: %d \n", (int)(ocl->workGroups));

  ocl->total_vel = clCreateBuffer(
	  ocl->context, CL_MEM_READ_WRITE,
	  sizeof(cl_float)*(ocl->workGroups)*params->maxIters, NULL, &err);
  checkError(err, "creating vel buffer", __LINE__);

  total_vel = (float*)malloc(sizeof(float)*(ocl->workGroups)*params->maxIters);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, float** cells_ptr, float** tmp_cells_ptr,
	short** obstacles_ptr, float** av_vels_ptr, t_ocl ocl)
{
  /*
  ** free up allocated memory
  */
  free(*cells_ptr);
  *cells_ptr = NULL;

  free(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  clReleaseMemObject(ocl.cells);
  clReleaseMemObject(ocl.tmp_cells);
  clReleaseMemObject(ocl.obstacles);
  clReleaseKernel(ocl.collision);

  clReleaseProgram(ocl.program);
  clReleaseCommandQueue(ocl.queue);
  clReleaseContext(ocl.context);

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, float* cells, short* obstacles, t_ocl ocl, float vel)
{
  const float viscosity = 1.0 / 6.0 * (2.0 / params.omega - 1.0);
  return  vel* params.reynolds_dim / viscosity;
}

float total_density(const t_param params, float* cells)
{
  float total = 0.0;  /* accumulator */

  for (int ii = 0; ii < params.ny; ii++)
  {
    for (int jj = 0; jj < params.nx; jj++)
    {
      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        total += cells[ INDEX(ii, jj, params.nx, params.ny, kk) ];
      }
    }
  }

  return total;
}

int write_values(const t_param params, float* cells, short* obstacles, float* av_vels)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.0 / 3.0; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.ny; ii++)
  {
    for (int jj = 0; jj < params.nx; jj++)
    {
      /* an occupied cell */
      if (obstacles[ii * params.nx + jj])
      {
        u_x = u_y = u = 0.0;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        local_density = 0.0;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[INDEX(ii, jj, params.nx, params.ny, kk)];
        }

        /* compute x velocity component */
        u_x = (cells[INDEX(ii, jj, params.nx, params.ny, 1)]
               + cells[INDEX(ii, jj, params.nx, params.ny, 5)]
               + cells[INDEX(ii, jj, params.nx, params.ny, 8)]
               - (cells[INDEX(ii, jj, params.nx, params.ny, 3)]
                  + cells[INDEX(ii, jj, params.nx, params.ny, 6)]
                  + cells[INDEX(ii, jj, params.nx, params.ny, 7)]))
              / local_density;
        /* compute y velocity component */
        u_y = (cells[INDEX(ii, jj, params.nx, params.ny, 2)]
               + cells[INDEX(ii, jj, params.nx, params.ny, 5)]
               + cells[INDEX(ii, jj, params.nx, params.ny, 6)]
               - (cells[INDEX(ii, jj, params.nx, params.ny, 4)]
                  + cells[INDEX(ii, jj, params.nx, params.ny, 7)]
                  + cells[INDEX(ii, jj, params.nx, params.ny, 8)]))
              / local_density;
        /* compute norm of velocity */
        u = sqrt((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", jj, ii, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void checkError(cl_int err, const char *op, const int line)
{
  if (err != CL_SUCCESS)
  {
    fprintf(stderr, "OpenCL error during '%s' on line %d: %d\n", op, line, err);
    fflush(stderr);
    exit(EXIT_FAILURE);
  }
}

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  getchar();
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}

#define MAX_DEVICES 32
#define MAX_DEVICE_NAME 1024

cl_device_id selectOpenCLDevice()
{
  cl_int err;
  cl_uint num_platforms = 0;
  cl_uint total_devices = 0;
  cl_platform_id platforms[8];
  cl_device_id devices[MAX_DEVICES];
  char name[MAX_DEVICE_NAME];

  // Get list of platforms
  err = clGetPlatformIDs(8, platforms, &num_platforms);
  checkError(err, "getting platforms", __LINE__);

  // Get list of devices
  for (cl_uint p = 0; p < num_platforms; p++)
  {
    cl_uint num_devices = 0;
    err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL,
                         MAX_DEVICES-total_devices, devices+total_devices,
                         &num_devices);
    checkError(err, "getting device name", __LINE__);
    total_devices += num_devices;
  }

  // Print list of devices
  printf("\nAvailable OpenCL devices:\n");
  for (cl_uint d = 0; d < total_devices; d++)
  {
    clGetDeviceInfo(devices[d], CL_DEVICE_NAME, MAX_DEVICE_NAME, name, NULL);
    printf("%2d: %s\n", d, name);
  }
  printf("\n");

  // Use first device unless OCL_DEVICE environment variable used
  cl_uint device_index = 0;
  char *dev_env = getenv("OCL_DEVICE");
  if (dev_env)
  {
    char *end;
    device_index = strtol(dev_env, &end, 10);
    if (strlen(end))
      die("invalid OCL_DEVICE variable", __LINE__, __FILE__);
  }

  if (device_index >= total_devices)
  {
    fprintf(stderr, "device index set to %d but only %d devices available\n",
            device_index, total_devices);
    exit(1);
  }

  // Print OpenCL device name
  clGetDeviceInfo(devices[device_index], CL_DEVICE_NAME,
                  MAX_DEVICE_NAME, name, NULL);
  printf("Selected OpenCL device:\n-> %s (index=%d)\n\n", name, device_index);

  return devices[device_index];
}
