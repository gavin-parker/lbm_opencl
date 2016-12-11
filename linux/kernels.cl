
#define NSPEEDS         9
#define C_SQ		(1.0 / 3.0) /* square of speed of sound */
#define W0			(4.0 / 9.0)  /* weighting factor */
#define W1			(W0 / 4)
#define W2			(W1 / 4)
#define C_SQ_2		(1.0 / ((C_SQ * C_SQ) + (C_SQ * C_SQ)))
#define C_SQ_R_2	(3.0 / 2.0)
#define INDEX(ii,jj,nx,ny,speed) (((nx)*(ny)*(speed))+ ((ii)*(nx)+(jj)))
#ifndef BLOCK_I
	#define BLOCK_I 16
#endif
#ifndef BLOCK_J
	#define BLOCK_J 16
#endif



kernel void accelerate_flow(global float* cells,
                            global short* obstacles,
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
      && (cells[INDEX(ii,jj,nx,ny,3)] - w1) > 0.0
      && (cells[INDEX(ii,jj,nx,ny,6)] - w2) > 0.0
      && (cells[INDEX(ii,jj,nx,ny,7)] - w2) > 0.0)
  {
    /* increase 'east-side' densities */
    cells[INDEX(ii,jj,nx,ny,1)] += w1;
    cells[INDEX(ii,jj,nx,ny,5)] += w2;
    cells[INDEX(ii,jj,nx,ny,8)] += w2;
    cells[INDEX(ii,jj,nx,ny,3)] -= w1;
    cells[INDEX(ii,jj,nx,ny,6)] -= w2;
    cells[INDEX(ii,jj,nx,ny,7)] -= w2;
  }
}

kernel void collision(global float* cells, global float* tmp_cells, global short* obstacles, int nx, int ny, float omega,  global float* tot_vel, int tt)
{
	local float scratch[64*2];

	int jj = get_global_id(0);
	int ii = get_global_id(1);

	int jj_local = get_local_id(0);
	int ii_local = get_local_id(1);

	int nx_local = get_local_size(0);
	int ny_local = get_local_size(1);
	float tot_u = 0;
	int local_index = ii_local*nx_local + jj_local;
	int local_size = nx_local * ny_local;
	int y_n = ((ii + 1) & (ny-1));
	int y_s = ((ii == 0) ? (ii + ny - 1) : (ii - 1));
	//wait until flow acceleration has been computed

	/*Propagate step*/
	/* determine indices of axis-direction neighbours
	** respecting periodic boundary conditions (wrap around) */
	int x_e = ((jj + 1) & (nx-1));
	int x_w = (jj == 0) ? (jj + nx - 1) : (jj - 1);
	/* propagate densities to neighbouring cells, following
	** appropriate directions of travel and writing into
	** scratch space grid */
	float speeds[9];
	speeds[0] = cells[INDEX(ii,jj,nx,ny,	0)];	
	speeds[1] = cells[INDEX(ii,x_w,nx,ny,	1)];
	speeds[3] = cells[INDEX(ii,x_e,nx,ny,	3)];
	speeds[5] = cells[INDEX(y_s,x_w,nx,ny,	5)];
	speeds[2] = cells[INDEX(y_s,jj,nx,ny,	2)];
	speeds[6] = cells[INDEX(y_s,x_e,nx,ny,	6)];
	speeds[8] = cells[INDEX(y_n,x_w,nx,ny,	8)];
	speeds[4] = cells[INDEX(y_n,jj,nx,ny,	4)];
	speeds[7] = cells[INDEX(y_n,x_e,nx,ny,	7)];
	int obstacle = !((int)obstacles[ii*nx + jj]);
	/* compute local density total */
	float local_density = 0.0;

	#pragma unroll
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

	u_x = native_divide(u_x,local_density);
	u_y = native_divide(u_y, local_density);
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
	tot_u = native_sqrt(u_sq)*obstacle;
	u_x  = native_divide(u_x, (float)C_SQ);
	u_y = native_divide(u_y, (float)C_SQ);
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
		u[0] = (speeds[0] + omega*u[0])*obstacle;
	float w1_local = W1 * local_density;
		#pragma unroll
	for (int i = 1; i < 5; i++) {
		float a = w1_local * (u[i]) - speeds[i];
		u[i] = speeds[i] + omega*a;
	}
	float w2_local = W2 * local_density;
	#pragma unroll
	for (int i = 5; i < 9; i++) {
		float a = w2_local * (u[i]) - speeds[i];
		u[i] = speeds[i] + omega*a;
	}
	if(!obstacle){
		u[1] = speeds[3];
		u[2] = speeds[4];
		u[3] = speeds[1];
		u[4] = speeds[2];
		u[5] = speeds[7];
		u[6] = speeds[8];
		u[7] = speeds[5];
		u[8] = speeds[6];
	}

       
	scratch[local_index] = tot_u;
	barrier(CLK_LOCAL_MEM_FENCE);
	#pragma unroll
	for(int i=0; i < 9 ; i++){ 
		tmp_cells[INDEX(ii,jj,nx,ny,i)] = u[i];
	}
	for(int offset = local_size/2; offset > 0; offset = offset / 2){
		if(local_index < offset){
			float other = scratch[local_index + offset];
			float mine = scratch[local_index];
			scratch[local_index] = mine + other;
		}
	barrier(CLK_LOCAL_MEM_FENCE);
	}
	if(local_index == 0){
		tot_vel[tt*(get_num_groups(0)*get_num_groups(1)) + (get_group_id(0) + get_group_id(1)*get_num_groups(0))] = scratch[0];
	}

		
}
