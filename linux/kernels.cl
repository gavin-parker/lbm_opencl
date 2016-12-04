
#define NSPEEDS         9
#define C_SQ		(1.0 / 3.0) /* square of speed of sound */
#define W0			(4.0 / 9.0)  /* weighting factor */
#define W1			(W0 / 4)
#define W2			(W1 / 4)
#define C_SQ_2		(1.0 / ((C_SQ * C_SQ) + (C_SQ * C_SQ)))
#define C_SQ_R_2	(3.0 / 2.0)
typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

kernel void accelerate_flow(global t_speed* cells,
                            global int* obstacles,
                            int nx, int ny,
                            float density, float accel)
{
  /* compute weighting factors */
  float w1 = density * accel / 9.0;
  float w2 = density * accel / 36.0;

  /* modify the 2nd row of the grid */
  int ii = ny - 2;

  /* get column index */
  int jj = get_global_id(0);

  /* if the cell is not occupied and
  ** we don't send a negative density */
  if (!obstacles[ii * nx + jj]
      && (cells[ii * nx + jj].speeds[3] - w1) > 0.0
      && (cells[ii * nx + jj].speeds[6] - w2) > 0.0
      && (cells[ii * nx + jj].speeds[7] - w2) > 0.0)
  {
    /* increase 'east-side' densities */
    cells[ii * nx + jj].speeds[1] += w1;
    cells[ii * nx + jj].speeds[5] += w2;
    cells[ii * nx + jj].speeds[8] += w2;
    /* decrease 'west-side' densities */
    cells[ii * nx + jj].speeds[3] -= w1;
    cells[ii * nx + jj].speeds[6] -= w2;
    cells[ii * nx + jj].speeds[7] -= w2;
  }
}

kernel void propagate(global t_speed* cells,
                      global t_speed* tmp_cells,
                      global int* obstacles,
                      int nx, int ny)
{
  /* get column and row indices */
  int jj = get_global_id(0);
  int ii = get_global_id(1);

  /* determine indices of axis-direction neighbours
  ** respecting periodic boundary conditions (wrap around) */
  int y_n = (ii + 1) % ny;
  int x_e = (jj + 1) % nx;
  int y_s = (ii == 0) ? (ii + ny - 1) : (ii - 1);
  int x_w = (jj == 0) ? (jj + nx - 1) : (jj - 1);
  /* propagate densities to neighbouring cells, following
  ** appropriate directions of travel and writing into
  ** scratch space grid */
  tmp_cells[ii  * nx + jj ].speeds[0] = cells[ii * nx + jj].speeds[0]; /* central cell, no movement */
  tmp_cells[ii  * nx + x_e].speeds[1] = cells[ii * nx + jj].speeds[1]; /* east */
  tmp_cells[y_n * nx + jj ].speeds[2] = cells[ii * nx + jj].speeds[2]; /* north */
  tmp_cells[ii  * nx + x_w].speeds[3] = cells[ii * nx + jj].speeds[3]; /* west */
  tmp_cells[y_s * nx + jj ].speeds[4] = cells[ii * nx + jj].speeds[4]; /* south */
  tmp_cells[y_n * nx + x_e].speeds[5] = cells[ii * nx + jj].speeds[5]; /* north-east */
  tmp_cells[y_n * nx + x_w].speeds[6] = cells[ii * nx + jj].speeds[6]; /* north-west */
  tmp_cells[y_s * nx + x_w].speeds[7] = cells[ii * nx + jj].speeds[7]; /* south-west */
  tmp_cells[y_s * nx + x_e].speeds[8] = cells[ii * nx + jj].speeds[8]; /* south-east */
}


kernel void collision(global t_speed* cells, global t_speed* tmp_cells, global int* obstacles, int nx, int ny, float omega,  global float* tot_vel)
{
	local float scratch[16*16];
	const float c_sq = 1.0 / 3.0; /* square of speed of sound */
	const float w0 = 4.0 / 9.0;  /* weighting factor */
	const float w1 = 1.0 / 9.0;  /* weighting factor */
	const float w2 = 1.0 / 36.0; /* weighting factor */
	int jj = get_global_id(0);
	int ii = get_global_id(1);

	int jj_local = get_local_id(0);
	int ii_local = get_local_id(1);

	int nx_local = get_local_size(0);
	int ny_local = get_local_size(1);
	float tot_u = 0;
	int local_index = ii_local*nx_local + jj_local;
	int local_size = nx_local * ny_local;
	int y_n = ((ii + 1) % ny)*nx;
	int y_s = ((ii == 0) ? (ii + ny - 1) : (ii - 1))*nx;
	int y = ii*nx;
	//wait until flow acceleration has been computed

	/*Propagate step*/
	/* determine indices of axis-direction neighbours
	** respecting periodic boundary conditions (wrap around) */
	int x_e = (jj + 1) % nx;
	int x_w = (jj == 0) ? (jj + nx - 1) : (jj - 1);
	/* propagate densities to neighbouring cells, following
	** appropriate directions of travel and writing into
	** scratch space grid */
	int current = y + jj;
	float speeds[9];
	speeds[0] = cells[current].speeds[0];
	speeds[1] = cells[y + x_w].speeds[1];
	speeds[3] = cells[y + x_e].speeds[3];
	speeds[5] = cells[y_s + x_w].speeds[5];
	speeds[2] = cells[y_s + jj].speeds[2];
	speeds[6] = cells[y_s + x_e].speeds[6];
	speeds[8] = cells[y_n + x_w].speeds[8];
	speeds[4] = cells[y_n + jj].speeds[4];
	speeds[7] = cells[y_n + x_e].speeds[7];
	int obstacle = !((int)obstacles[current]);
	/* compute local density total */
	float local_density = 0.0;


	for (int i = 0; i < 9; i++) {
		local_density += speeds[i];
	}

	float a = speeds[8] - speeds[6];
	float b = speeds[5] - speeds[7];
	float c = speeds[1] - speeds[3];
	float d = speeds[2] - speeds[4];
	/* compute x velocity component */
	float u_x = b;
	float u_y = b;
	u_x = u_x + a + c;
	u_y = u_y + d - a;

	u_x /= local_density;
	u_y /= local_density;
	float w_local = W0 * local_density;
	/* velocity squared */

	/* directional velocity components */


	//Compiler can you please vectorize this ty xoxo

	//before: 9, now:
	float u_x_y = u_x * u_y * 2.0 * C_SQ_2;
	float u2[NSPEEDS];
	u2[1] = u_x * u_x;
	u2[2] = u_y * u_y;
	float u_sq = (u2[1] + u2[2]);
	float u_sq_recip = u_sq *C_SQ_R_2 - 1.0;

	u2[0] = 0;
	u2[1] = u2[1] * C_SQ_2 - u_sq_recip;
	u2[2] *= C_SQ_2;
	u2[3] = u2[1];
	u2[4] = u2[2] - u_sq_recip;
	u2[5] = u2[1] + u2[2] + u_x_y;
	u2[6] = u2[1] + u2[2] - u_x_y;
	u2[7] = u2[5];
	u2[8] = u2[6];
	u2[2] -= u_sq_recip;
	tot_u = sqrt(u_sq)*obstacle;
	u_x /= C_SQ;
	u_y /= C_SQ;
	float u[NSPEEDS];
	u[0] = -u_sq_recip;
	u[1] = u_x + u2[1];
	u[2] = u_y + u2[2];
	u[3] = -u_x + u2[3];
	u[4] = -u_y + u2[4];
	u[5] = u_x + u_y + u2[5];
	u[6] = -u_x + u_y + u2[6];
	u[7] = -u_x - u_y + u2[7];
	u[8] = u_x - u_y + u2[8];

	u[0] = w_local * (u[0]) - speeds[0];
		tmp_cells[current].speeds[0] = (speeds[0] + omega*u[0])*obstacle;
	float w1_local = W1 * local_density;
	for (int i = 1; i < 5; i++) {
		float a = w1_local * (u[i]) - speeds[i];
		tmp_cells[current].speeds[i] = speeds[i] + omega*a;
	}
	float w2_local = W2 * local_density;
	for (int i = 5; i < 9; i++) {
		float a = w2_local * (u[i]) - speeds[i];
		tmp_cells[current].speeds[i] = speeds[i] + omega*a;
	}
       
	scratch[local_index] = tot_u;
	barrier(CLK_LOCAL_MEM_FENCE);
	for(int offset = local_size/2; offset > 0; offset = offset / 2){
		if(local_index < offset){
			float other = scratch[local_index + offset];
			float mine = scratch[local_index];
			scratch[local_index] = mine + other;
		}
	barrier(CLK_LOCAL_MEM_FENCE);
	}
	if(local_index == 0){
		tot_vel[get_group_id(0) + get_group_id(1)*get_num_groups(0)] = scratch[0];
	}

		
}

kernel void rebound(global t_speed* cells,global t_speed* tmp_cells, global int* obstacles, int nx, int ny)
{

  int jj = get_global_id(0);
  int ii = get_global_id(1);

      /* if the cell contains an obstacle */
      if (obstacles[ii * nx + jj])
      {
		int y_n = ((ii + 1) % ny)*nx;
		int y_s = ((ii == 0) ? (ii + ny - 1) : (ii - 1))*nx;
		int y = ii*nx;
		int current = y + jj;

		int x_e = (current + 1) % nx;
		int x_w = (jj == 0) ? (jj + nx - 1) : (jj - 1);
		tmp_cells[current].speeds[1] = cells[y + x_e].speeds[3];
		tmp_cells[current].speeds[2] = cells[y_n + jj].speeds[4];
		tmp_cells[current].speeds[3] = cells[y + x_w].speeds[1];
		tmp_cells[current].speeds[4] = cells[y_s + jj].speeds[2];
		tmp_cells[current].speeds[5] = cells[y_n + x_e].speeds[7];
		tmp_cells[current].speeds[6] = cells[y_n + x_w].speeds[8];
		tmp_cells[current].speeds[7] = cells[y_s + x_w].speeds[5];
		tmp_cells[current].speeds[8] = cells[y_s + x_e].speeds[6];
      }
}


kernel void av_velocity(global t_speed* cells, global int* obstacles, global float* tot_vel, int nx)
{


  local float scratch[16*16];
       /* accumulated magnitudes of velocity for each cell */
  int jj = get_global_id(0);
  int ii = get_global_id(1);

  int jj_local = get_local_id(0);
  int ii_local = get_local_id(1);

  int nx_local = get_local_size(0);
  int ny_local = get_local_size(1);
  float tot_u = 0;
  int local_index = ii_local*nx_local + jj_local;
  int local_size = nx_local * ny_local;
  /* loop over all non-blocked cells */

      /* ignore occupied cells */
      if (!obstacles[ii * nx + jj])
      {
        /* local density total */
        float local_density = 0.0;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii * nx + jj].speeds[kk];
        }

        /* x-component of velocity */
        float u_x = (cells[ii * nx + jj].speeds[1]
                      + cells[ii * nx + jj].speeds[5]
                      + cells[ii * nx + jj].speeds[8]
                      - (cells[ii * nx + jj].speeds[3]
                         + cells[ii * nx + jj].speeds[6]
                         + cells[ii * nx + jj].speeds[7]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (cells[ii * nx + jj].speeds[2]
                      + cells[ii * nx + jj].speeds[5]
                      + cells[ii * nx + jj].speeds[6]
                      - (cells[ii * nx + jj].speeds[4]
                         + cells[ii * nx + jj].speeds[7]
                         + cells[ii * nx + jj].speeds[8]))
                     / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u = sqrt((u_x * u_x) + (u_y * u_y));

      }

	   
	  scratch[local_index] = tot_u;
	  	barrier(CLK_LOCAL_MEM_FENCE);
	  for(int offset = local_size/2; offset > 0; offset = offset / 2){
		if(local_index < offset){
			float other = scratch[local_index + offset];
			float mine = scratch[local_index];
			scratch[local_index] = mine + other;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	  }
	  if(local_index == 0){
		tot_vel[get_group_id(0) + get_group_id(1)*get_num_groups(0)] = scratch[0];
		//printf("local_size: %d\n", local_size);
	  }



	//printf("tot_vel: %f\n", *tot_vel);

}