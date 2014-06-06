/*
QuadTree N BODY

QuadTree Encoding
Thrust Sort
Compute Center of Mass
NBody
	Body Interactions for region
	Body Interactions for neighbouring regions
	Far-field approximations using center of mass

Possible Improvements
	Tiling using Shared Memory
	Use constant memory for LUT
*/
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cmath>
#include <thrust/sort.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <numeric>
#define _USE_MATH_DEFINES
#include "math.h"

#ifdef _WIN32
#include<windows.h>
#endif
#include <GL/glew.h>
#include <GL/wglew.h>
#include <GL/glut.h>

const int N = 512;		// Number of Bodies
const int n = 2;		// Regions are 2^n * 2^n
int ThreadsQ = 128;		// blocksize for createquadtree
int ThreadsX = 128;		// blocksize for interactions kernel
int ThreadsC = 32; // blocksize for compute com

const int iterations = 100;	// Number of Interations for benchmarking
const double sim_rad = 1e18;// Simulation Radius

// Benchmarking vectors
std::vector<float> results;
std::vector<float> vresults;
std::vector<float> sresults;
std::vector<float> cresults;

// Description of Quad
struct Quad
{
	double2 center;
	double w;
	__device__ Quad(double2 _c, double _w)
	{
		center = _c;
		w = _w;
	}
	void print()
	{
		// Center (x, y) width:
		std::cout << "Center (" << center.x << ", " << center.y << ") width: " << w << std::endl;
	}
};

int* LUT_d;
int4* index_d;
bool benchmark;

// Description of Body
struct body
{
	float4 colour;
	double4 position;
	double4 velocity;
	int morton;
	bool operator==(body b)
	{
		return( (position.x == b.position.x) && (position.y == b.position.y) && (velocity.x == b.velocity.x) && (velocity.y == b.velocity.y) );
	}
	void print()
	{
		std::cout << "\tMorton : " << morton;
		std::cout << "\n\tPosition(x,y): " << position.x << " " << position.y;
		std::cout << "\n\tV: " << velocity.x << " " << velocity.y;
		std::cout << "\tF: " << velocity.z << " " << velocity.w;
		std::cout << "\n\tMass: " << position.w << std::endl << std::endl;
	}
};

// sort Morton
__host__ __device__ bool operator<(const body &lhs, const body &rhs) 
{
	return lhs.morton < rhs.morton;
}

// Body initialisation functions
double circlev(double x, double y)
{
	double solarmass = 1.98892e30;
	double r2 = sqrt(x*x + y*y);
	double numerator = (6.67e-11)*1e6*solarmass;
	return sqrt(numerator/r2);
}
double random()			{ return ((double)rand())/((double)RAND_MAX); }
template <typename T> int signum(T val) {
    return (T(0) < val) - (val < T(0));
}
body init_body(int i)
{
	double solarmass = 1.98892e30;
	if(i != 0)
	{
		double px = sim_rad*exp(-1.8)*(0.5 - random());
		double py = sim_rad*exp(-1.8)*(0.5 - random());
		double magv = circlev(px, py);

		double absangle = atan(abs(py/px));
		double thetav = M_PI/2 - absangle;
		double phiv = random() * M_PI;
		double vx = -1*signum(py)*cos(thetav)*magv;
		double vy = signum(px)*sin(thetav)*magv;

		if(random() < 0.5)
		{
			vx = -vx;
			vy = -vy;
		}

		double mass = random() * solarmass*10+1e20;

		body b;
		b.position.x = px;
		b.position.y = py;
		b.position.z = 0;
		b.position.w = mass;

		b.velocity.x = vx;
		b.velocity.y = vy;
		b.velocity.z = 0;
		b.velocity.w = 0;

		b.colour.x = 1.0f;
		b.colour.y = 0.0f;
		b.colour.z = 0.0f;
		b.colour.w = 1.0f;

		b.morton = 0;
		return b;
	}
	else
	{
		body b;
		b.position.x = 0;
		b.position.y = 0;
		b.position.z = 0;
		b.position.w = 1e6*solarmass;

		b.velocity.x = 0;
		b.velocity.y = 0;
		b.velocity.z = 0;
		b.velocity.w = 0;

		b.colour.x = 1.0f;
		b.colour.y = 1.0f;
		b.colour.z = 0.0f;
		b.colour.w = 1.0f;

		b.morton = 0;
		return b;
	}
}

	body* b_in;
	body* b_out;
	body* b_draw;
	body* display_bodies;

void cudaQuery();

__device__ double2 bodyBodyInteraction(double4 bi, double4 bj, double2 a, bool output_thread)
{
    double G = 6.67e-11;   // gravational constant
    double EPS = 3E4;      // softening parameter

	// [2 FLOPS] 
	double dx = bj.x - bi.x;
	double dy = bj.y - bi.y;

	// [5 FLOPS]
	double dist = sqrt(dx*dx + dy*dy) + 0.0000125; // additional softening parameter
	
	// [6 FLOPS]
	double F = (G * bi.w * bj.w) / (dist*dist + EPS*EPS);

	// [6 FLOPS]
	a.x += F * dx / dist;
	a.y += F * dy / dist;

	return a;
}

__device__ int findFirstValue(body* body_in, int h)
{
	int left = -1;
	int right = N-1;

	while(right > left + 1)
	{
		int middle = (left + right)/2;
		if(body_in[middle].morton >= h)
			right = middle;
		else
			left = middle;
	}

	if(right < N && body_in[right].morton == h)
		return right;

	return -1;
}

__global__ void
compute_center_mass_with_search(body* body_in, int4* index_store, int lim)
{
    //	idx( 0 to number of regions )
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx > lim )
		return;

	bool mass = 0;
	int first = 0;
	int last = 0;
	int total = 0;

	int start = findFirstValue(body_in, idx);

	if(start != -1)
	{
		// for all bodies
		for(int b = start; b < N; b++)
		{
			int hil = body_in[b].morton;
			if(idx == hil)
			{
				// if first hilbert in list
				if(!mass)
				{
					first = b;
					mass = true;					
				}

				total++;
			}
			if(hil > idx)
			{
				first = (mass == 0) ? b : first;
	
				last = b;
				break;
			}
		}
	}

	last = ((last == 0) && (first != 0)) ? N : last;

	//center of mass for voronoi region
	index_store[idx].x = first;
	index_store[idx].y = last;
	index_store[idx].z = total;
}

__device__ double2 computeHilbertBodies(int h, int4* index_store, body* body_in, double4 position, double2 force)
{
	int4 index = index_store[h];
	for(int i = index.x; i < index.y; i++)
	{
		double4 p = body_in[i].position;
		force = bodyBodyInteraction(position, p, force, false);
	}

	return force;
}

__device__ int2 findValue(int* LUT, int h)
{
	int DIM = 1 << n;

	for(int y = 0; y < DIM+2; y++)
	{
		for(int x = 0; x < DIM +2; x++)
		{
			if(LUT[y*(DIM+2) + x] == h)
			{
				int2 coords;
				coords.x = x - 1;
				coords.y = y - 1;
				return coords;
			}
		}
	}

	int2 err;
	err.x = -1;
	err.y = -1;
	return err;
}

// "Insert" a 0 bit after each of the 16 low bits of x
__device__ __host__ int Part1By1(int x)
{
	x &= 0x0000ffff;                  // x = ---- ---- ---- ---- fedc ba98 7654 3210
	x = (x ^ (x <<  8)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
	x = (x ^ (x <<  4)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
	x = (x ^ (x <<  2)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
	x = (x ^ (x <<  1)) & 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
	
	return x;
}

__device__ __host__ int EncodeMorton2(int x, int y)
{
	return (Part1By1(y) << 1) + Part1By1(x);
}

// Inverse of Part1By1 - "delete" all odd-indexed bits
__device__ int Compact1By1(int x)
{
  x &= 0x55555555;                  // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
  x = (x ^ (x >>  1)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
  x = (x ^ (x >>  2)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
  x = (x ^ (x >>  4)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
  x = (x ^ (x >>  8)) & 0x0000ffff; // x = ---- ---- ---- ---- fedc ba98 7654 3210
  return x;
}

__device__ int DecodeMorton2X(int code)
{
  return Compact1By1(code >> 0);
}

__device__ int DecodeMorton2Y(int code)
{
  return Compact1By1(code >> 1);
}

__device__ int Encode(int x, int y, int DIM)
{
	int code;

	//check boundary
	if((x < 0) || (x > (DIM - 1)) || (y < 0) || (y > (DIM - 1)) )
		code = -1;
	else 
		code = EncodeMorton2(x, y);

	return code;
}

// Buffer Area Usage calculating codes
__global__ void 
BA_kernel(body* body_draw)
{
	// index is input region
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

	// return if more than number of regions available
	if(idx > 1<<2*n)
		return;

	// DIM = 2^n (grid is 2^n * 2^n)
	int DIM = 1 << n;

	// decode the index into x and y values
	int2 coords;
	coords.x = DecodeMorton2X(idx);
	coords.y = DecodeMorton2Y(idx);

	// encode neighbouring elements
	int regions[9];
	regions[0] = Encode(coords.x - 1,	coords.y - 1,	DIM);
	regions[1] = Encode(coords.x,		coords.y - 1,	DIM);
	regions[2] = Encode(coords.x + 1,	coords.y - 1,	DIM);

	regions[3] = Encode(coords.x - 1,	coords.y,		DIM);
	regions[4] = idx;
	regions[5] = Encode(coords.x + 1,	coords.y,		DIM);

	regions[6] = Encode(coords.x - 1,	coords.y + 1,	DIM);
	regions[7] = Encode(coords.x,		coords.y + 1,	DIM);
	regions[8] = Encode(coords.x + 1,	coords.y + 1,	DIM);

	// for identified regions
	for(int i = 0; i < 9; i++)
	{
		// if region is not out of bounds
		if(regions[i] > -1)
		{
			float4 colour;
			colour.x = 1.0f;
			colour.y = 1.0f;	
			colour.z = 1.0f;	
			colour.w = 1.0f;

			// alter output colour
			body_draw[idx * (1<<2*n) + regions[i]].colour = colour;
		}
	}
}

// Buffer Area Usage using LUT
__global__ void 
buffer_area_kernel(body* body_draw, int* LUT)
{
	// idx is region ID (vertical)
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx > 1<<2*n)
		return;

	int DIM = 1 << n;

	int2 coords = findValue(LUT, idx);
	
	// neighbouring region IDs (horizontal)
	int regions[9];
	regions[0] = LUT[ (coords.y) * (DIM+2) + coords.x     ];
	regions[1] = LUT[ (coords.y) * (DIM+2) + coords.x + 1 ];
	regions[2] = LUT[ (coords.y) * (DIM+2) + coords.x + 2 ];
	
	regions[3] = LUT[ (coords.y + 1) * (DIM+2) + coords.x     ];
	regions[4] = LUT[ (coords.y + 1) * (DIM+2) + coords.x + 1 ];
	regions[5] = LUT[ (coords.y + 1) * (DIM+2) + coords.x + 2 ];
	
	regions[6] = LUT[ (coords.y + 2) * (DIM+2) + coords.x     ];
	regions[7] = LUT[ (coords.y + 2) * (DIM+2) + coords.x + 1 ];
	regions[8] = LUT[ (coords.y + 2) * (DIM+2) + coords.x + 2 ];

	// each idx creates output for line
	for(int i = 0; i < 9; i++)
	{
		if(regions[i] > -1)
		{
			float4 colour;
			colour.x = 1.0f;
			colour.y = 1.0f;	
			colour.z = 1.0f;	
			colour.w = 1.0f;

			//change colour of output colour
			body_draw[idx * (1<<2*n) + regions[i]].colour = colour;
		}
	}
}

__global__ void 
nbody_kernel_LUT(body* body_in, int* LUT, int4* index_store, body* body_out)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx > N)
		return;

	float4 colour;
	colour.x = 0.0f;
	colour.y = 1.0f;	
	colour.z = 0.0f;	
	colour.w = 1.0f;	
	
	double4 position = body_in[idx].position;
	int morton = body_in[idx].morton;

	double2 force;
	force.x = 0.0;
	force.y = 0.0;

	if(morton > 0)
	{
		int DIM = 1 << n;

		// Indentify the regions neighbouring body[idx]
		int2 coords = findValue(LUT, morton);
	
		int regions[9];
		regions[0] = LUT[ (coords.y) * (DIM+2) + coords.x     ];
		regions[1] = LUT[ (coords.y) * (DIM+2) + coords.x + 1 ];
		regions[2] = LUT[ (coords.y) * (DIM+2) + coords.x + 2 ];
	   
		regions[3] = LUT[ (coords.y + 1) * (DIM+2) + coords.x     ];
		regions[4] = LUT[ (coords.y + 1) * (DIM+2) + coords.x + 1 ];
		regions[5] = LUT[ (coords.y + 1) * (DIM+2) + coords.x + 2 ];
	   
		regions[6] = LUT[ (coords.y + 2) * (DIM+2) + coords.x     ];
		regions[7] = LUT[ (coords.y + 2) * (DIM+2) + coords.x + 1 ];
		regions[8] = LUT[ (coords.y + 2) * (DIM+2) + coords.x + 2 ];

		//find the total number of bodies in regions above
		// this would involve reading the index store
		// as the index store is an int4 and uses only 3
		// for the first, last, and total bodies in the 
		// region, this could also be used to store the
		// LUT value and thus we could coalesce into 1 read
		/*
		int totalBodies = 0;
		int4 index[9];
		for(int i = 0; i < 9; i++)
		{
			if(regions[i] > -1)
			{
				index[i] = index_store[regions[i]];
				totalBodies += index[i].z;
			}
		}
		*/

		// calculate force contributions for body
		// for bodies in same region do body body interaction
		for(int i = 0; i < 9; i++)
		{
			if(regions[i] > -1)
				force = computeHilbertBodies(regions[i], index_store, body_in, position, force); 
		}

		colour = body_in[idx].colour;
	}

	// Do update
	double4 velocity = body_in[idx].velocity;
	double2 v;
	v.x = velocity.x + 1e10 * force.x / position.w;
	v.y = velocity.y + 1e10 * force.y / position.w;

	body_out[idx].velocity.x = v.x;
	body_out[idx].velocity.y = v.y;
	body_out[idx].position.x = position.x + 1e10 * v.x;
	body_out[idx].position.y = position.y + 1e10 * v.y;
	body_out[idx].position.w = position.w;
	body_out[idx].colour = colour;
	body_out[idx].morton = body_in[idx].morton;
}

__global__ void 
nbody_kernel_CODE(body* body_in, int4* index_store, body* body_out)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx > N)
		return;

	float4 colour;
	colour.x = 0.0f;
	colour.y = 1.0f;	
	colour.z = 0.0f;	
	colour.w = 1.0f;	
	
	double4 position = body_in[idx].position;
	int morton = body_in[idx].morton;

	double2 force;
	force.x = 0.0;
	force.y = 0.0;

	if(morton > 0)
	{
		int DIM = 1 << n;

		// Indentify the regions neighbouring body[idx]
		int2 coords;
		coords.x = DecodeMorton2X(morton);
		coords.y = DecodeMorton2Y(morton);
	
		/*
		// encode neighbouring elements
		int regions[9];
		regions[0] = Encode(coords.x - 1,	coords.y - 1,	DIM);
		regions[1] = Encode(coords.x,		coords.y - 1,	DIM);
		regions[2] = Encode(coords.x + 1,	coords.y - 1,	DIM);

		regions[3] = Encode(coords.x - 1,	coords.y,		DIM);
		regions[4] = morton;
		regions[5] = Encode(coords.x + 1,	coords.y,		DIM);

		regions[6] = Encode(coords.x - 1,	coords.y + 1,	DIM);
		regions[7] = Encode(coords.x,		coords.y + 1,	DIM);
		regions[8] = Encode(coords.x + 1,	coords.y + 1,	DIM);

		// calculate force contributions for body
		// for bodies in same region do body body interaction
		for(int i = 0; i < 9; i++)
		{
			if(regions[i] > -1)
				force = computeHilbertBodies(regions[i], index_store, body_in, position, force); 
		}
		*/

		for(int i = -1; i < 2; i++)
		{
			for(int j = -1; j < 2; j++)
			{
				int region = Encode(coords.x + i, coords.y + j, DIM);
				if(region > -1)
					force = computeHilbertBodies(region, index_store, body_in, position, force); 
			}
		}

		colour = body_in[idx].colour;
	}

	// Do update
	double4 velocity = body_in[idx].velocity;
	double2 v;
	v.x = velocity.x + 1e10 * force.x / position.w;
	v.y = velocity.y + 1e10 * force.y / position.w;

	body_out[idx].velocity.x = v.x;
	body_out[idx].velocity.y = v.y;
	body_out[idx].position.x = position.x + 1e10 * v.x;
	body_out[idx].position.y = position.y + 1e10 * v.y;
	body_out[idx].position.w = position.w;
	body_out[idx].colour = colour;
	body_out[idx].morton = body_in[idx].morton;
}

__global__ void 
nbody_kernel_groups(body* body_in, int* LUT, int4* index_store, body* body_out)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx > N)
		return;

	float4 colour;
	colour.x = 0.0f;
	colour.y = 1.0f;	
	colour.z = 0.0f;	
	colour.w = 1.0f;	
	
	double4 position = body_in[idx].position;
	int morton = body_in[idx].morton;

	double2 force;
	force.x = 0.0;
	force.y = 0.0;

	if(morton > 0)
	{
		int DIM = 1 << n;

		// Indentify the regions neighbouring body[idx]
		int2 coords = findValue(LUT, morton);
	
		int regions[9];
		regions[0] = LUT[ (coords.y) * (DIM+2) + coords.x     ];
		regions[1] = LUT[ (coords.y) * (DIM+2) + coords.x + 1 ];
		regions[2] = LUT[ (coords.y) * (DIM+2) + coords.x + 2 ];
	   
		regions[3] = LUT[ (coords.y + 1) * (DIM+2) + coords.x     ];
		regions[4] = LUT[ (coords.y + 1) * (DIM+2) + coords.x + 1 ];
		regions[5] = LUT[ (coords.y + 1) * (DIM+2) + coords.x + 2 ];
	   
		regions[6] = LUT[ (coords.y + 2) * (DIM+2) + coords.x     ];
		regions[7] = LUT[ (coords.y + 2) * (DIM+2) + coords.x + 1 ];
		regions[8] = LUT[ (coords.y + 2) * (DIM+2) + coords.x + 2 ];

		//find the total number of bodies in regions above
		// this would involve reading the index store
		// as the index store is an int4 and uses only 3
		// for the first, last, and total bodies in the 
		// region, this could also be used to store the
		// LUT value and thus we could coalesce into 1 read
		/*
		int totalBodies = 0;
		int4 index[9];
		for(int i = 0; i < 9; i++)
		{
			if(regions[i] > -1)
			{
				index[i] = index_store[regions[i]];
				totalBodies += index[i].z;
			}
		}
		*/

		// calculate force contributions for body
		// for bodies in same region do body body interaction
		for(int i = 0; i < 9; i++)
		{
			if(regions[i] > -1)
				force = computeHilbertBodies(regions[i], index_store, body_in, position, force); 
		}

		colour = body_in[idx].colour;
	}

	// Do update
	double4 velocity = body_in[idx].velocity;
	double2 v;
	v.x = velocity.x + 1e10 * force.x / position.w;
	v.y = velocity.y + 1e10 * force.y / position.w;

	body_out[idx].velocity.x = v.x;
	body_out[idx].velocity.y = v.y;
	body_out[idx].position.x = position.x + 1e10 * v.x;
	body_out[idx].position.y = position.y + 1e10 * v.y;
	body_out[idx].position.w = position.w;
	body_out[idx].colour = colour;
	body_out[idx].morton = body_in[idx].morton;
}

__device__ int getQuadID(double2 p, Quad& q)
{
	int ID = -1;
	double2 c;
	if(p.x < q.center.x)	// 0 or 2
	{
		if(p.y < q.center.y)
		{		
			c.x = q.center.x - 0.25*q.w;
			c.y = q.center.y - 0.25*q.w;
			ID = 2;
		}
		else
		{
			c.x = q.center.x - 0.25*q.w;
			c.y = q.center.y + 0.25*q.w;
			ID = 0;
		}
	}
	else			// 1 or 3
	{
		if(p.y < q.center.y)
		{
			c.x = q.center.x + 0.25*q.w;
			c.y = q.center.y - 0.25*q.w;
			ID = 3;
		}
		else
		{
			c.x = q.center.x + 0.25*q.w;
			c.y = q.center.y + 0.25*q.w;
			ID = 1;
		}
	}

	// update quad
	q = Quad( c, 0.5*q.w );	
	return ID;
}

__device__ bool containsPoint(double2 p, double radius)
{
	if(p.x >= -1*radius &&
		p.x <= radius &&
		p.y >= -1*radius &&
		p.y <= radius)
		return true;

	return false;		
}

// Create linear quad tree (morton) code kernel
__global__ void create_quadtree( body* body_in, double radius)
{
    // map from thread to pixel position
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// read body position
	double2 position;
	position.x = body_in[idx].position.x;
	position.y = body_in[idx].position.y;

	if(!containsPoint(position, radius))
	{
		body_in[idx].morton = -1;
		return;
	}

	// initial quad
	double2 c;
	c.x = 0.0;
	c.y = 0.0;
	Quad q( c, radius*2 );

	// generate code for zone
	int word = 0;
	int mult = 1 <<2*(n-1);	//4^(n-1)
	for(int i = 0; i < n; i++)
	{
		int ID = getQuadID( position, q );
		word += ID*mult;
		mult = mult >> 2; // divide by 4
	}

	// now save the generated code for that position
	body_in[idx].morton = word;
}

void cudaCheckAPIError(cudaError_t err, const char* file, int line)
{
	if(err != cudaSuccess)
	{
		std::stringstream ss;
		ss << file << "(" << line << ")";
		std::string file_and_line;
		ss >> file_and_line;
		throw thrust::system_error(err, thrust::cuda_category(), file_and_line);
	}
}

template <class T>
void printLine(const char* message, T value)
{
	std::cout << message << "\t : " << value << std::endl;
}
template <class T>
void printLine(const char* message, T* value)
{
	if(value[2] == NULL)
		std::cout << message << "\t : " << value[0] << ", " << value[1] << std::endl;
	else
		std::cout << message << "\t : " << value[0] << " " << value[1] << " " << value[2] << std::endl;
}

void printBlank()
{
	std::cout << std::endl;
}

size_t RoundUp(int groupSize, int globalSize)
{
    int r = globalSize % groupSize;
    if(r == 0)
    {
        return globalSize;
    }
    else
    {
        return globalSize + groupSize - r;
    }
}

void cleanup()
{
	//cleanup
	cudaCheckAPIError( cudaFree( b_in ), "kernel.cu", __LINE__  );
	cudaCheckAPIError( cudaFree( b_out ), "kernel.cu", __LINE__  );
	cudaCheckAPIError( cudaFree( b_draw ), "kernel.cu", __LINE__  );
	cudaCheckAPIError( cudaFree( LUT_d ), "kernel.cu", __LINE__  );
	cudaCheckAPIError( cudaFree( index_d), "kernel.cu", __LINE__  );

	free( display_bodies );

	exit(EXIT_SUCCESS);
}

void Key(unsigned char key, int x, int y)
{
    switch(key) 
    {
        case '\033': // escape quits
        case '\015': // Enter quits    
        case 'Q':    // Q quits
        case 'q':    // q (or escape) quits
            // Cleanup up and quit
                cleanup();
            break;
		case 'b':
			benchmark = true;
			break;
		case 'p':
			system("pause");
			break;
    }
}

float completeEvent(cudaEvent_t start, cudaEvent_t stop)
{
	// Add the stop event to the GPUs queue of work
	cudaCheckAPIError( cudaEventRecord(stop, 0), "kernel.cu", __LINE__  );
	
	// Wait until the event has completed so it is safe to read
	cudaCheckAPIError( cudaEventSynchronize(stop), "kernel.cu", __LINE__  );
	
	// Determine the time elapsed between the events
	float milliseconds = 0;
	cudaCheckAPIError( cudaEventElapsedTime(&milliseconds, start, stop), "kernel.cu", __LINE__  );

	return milliseconds;
}

void renderBodies(body* b)
{
	body bodies[N];

	// copy data from device to host
	//cudaCheckAPIError( cudaEventRecord(startEvent, 0) );
		cudaCheckAPIError( cudaMemcpy( bodies, b, sizeof(body)*N, cudaMemcpyDeviceToHost ), "kernel.cu", __LINE__  );
	//completeEvent(startEvent, stopEvent, "retrieving output", false);

	//for(int i = 0; i < N; i++)
	//	bodies[i].print();

    glEnable( GL_POINT_SMOOTH );
    glEnable( GL_BLEND );
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
    glPointSize( 2.0 );

	glClearColor( 0.0, 0.0, 1.0, 1.0 );
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//glColor3f(1, 0, 0);

	glBegin(GL_POINTS);
		for(int i = 0; i < N; i++)
		{
			glColor3f( bodies[i].colour.x, bodies[i].colour.y, bodies[i].colour.z );	
			glVertex2f(bodies[i].position.x/(sim_rad/4), bodies[i].position.y/(sim_rad/4));
		}
	glEnd();

	glFinish();
	glutSwapBuffers();

	//for(int i = 0; i < N; i++)
	//	bodies[i].print();

	//system("pause");
}

void outputCOM()
{
	int num = 1<<2*n;
	int4* index_store = new int4[num];

	// copy data from device to host
	cudaCheckAPIError( cudaMemcpy( index_store, index_d, sizeof(int4)*num, cudaMemcpyDeviceToHost ), "kernel.cu", __LINE__  );

	int tot = 0;
	for(int i = 0; i < num; i++)
	{
		std::cout <<"\tFirst" << index_store[i].x << " Last: " << index_store[i].y << " Zone Amount: " << index_store[i].z <<  std::endl;
		tot += index_store[i].z;
	}

	std::cout << "Total number of bodies accounted for " << tot << std::endl;

	delete [] index_store;
}

void bufferAreaKernel()
{
	// Event parameters 
	cudaEvent_t startEvent, stopEvent;

	// Create the event using cudaEventCreate
	cudaCheckAPIError( cudaEventCreate(&startEvent), "kernel.cu", __LINE__  );
	cudaCheckAPIError( cudaEventCreate(&stopEvent), "kernel.cu", __LINE__  );

	int gridSize = (1<<2*n)/32;
	if(gridSize < 1)
		gridSize = 1;
	dim3 grid( gridSize );
	dim3 block(32);

	cudaFuncSetCacheConfig(BA_kernel, cudaFuncCachePreferL1);

	// compute body updates
	cudaCheckAPIError( cudaEventRecord(startEvent, 0), "kernel.cu", __LINE__  );
		BA_kernel <<< grid, block >>> (b_draw );
	completeEvent(startEvent, stopEvent) ;

	// Release events
	cudaCheckAPIError( cudaEventDestroy(startEvent), "kernel.cu", __LINE__  );
	cudaCheckAPIError( cudaEventDestroy(stopEvent), "kernel.cu", __LINE__  );
}

// Execute voronoi kernel
void executeQuadTree()
{
	// Event parameters 
	cudaEvent_t startEvent, stopEvent;

	// Create the event using cudaEventCreate
	cudaCheckAPIError( cudaEventCreate(&startEvent), "kernel.cu", __LINE__  );
	cudaCheckAPIError( cudaEventCreate(&stopEvent), "kernel.cu", __LINE__  );

	int gridSize = N/(ThreadsQ);
	if(gridSize < 1)
		gridSize = 1;

	dim3 grid( gridSize );
	dim3 block( ThreadsQ );

	cudaFuncSetCacheConfig(create_quadtree, cudaFuncCachePreferL1);

	cudaCheckAPIError( cudaEventRecord(startEvent, 0), "kernel.cu", __LINE__  );
		create_quadtree <<< grid, block >>> (b_in, sim_rad/8);
	vresults.push_back( completeEvent(startEvent, stopEvent) );

	// Release events
	cudaCheckAPIError( cudaEventDestroy(startEvent), "kernel.cu", __LINE__  );
	cudaCheckAPIError( cudaEventDestroy(stopEvent), "kernel.cu", __LINE__  );
}

// compute center of mass
void computeCOM()
{
	// Event parameters 
	cudaEvent_t startEvent, stopEvent;

	// Create the event using cudaEventCreate
	cudaCheckAPIError( cudaEventCreate(&startEvent), "kernel.cu", __LINE__  );
	cudaCheckAPIError( cudaEventCreate(&stopEvent), "kernel.cu", __LINE__  );

	int totalSize = 1<<2*n;
	int gridSize = totalSize/ThreadsC;
	if (gridSize < 1)
		gridSize = 1;

	dim3 grid( gridSize );
	dim3 block( ThreadsC );

	cudaFuncSetCacheConfig(compute_center_mass_with_search, cudaFuncCachePreferL1);

	cudaCheckAPIError( cudaEventRecord(startEvent, 0), "kernel.cu", __LINE__  );
		compute_center_mass_with_search <<< grid, block >>> (b_out, index_d, totalSize);
	cresults.push_back( completeEvent(startEvent, stopEvent) );

	// Release events
	cudaCheckAPIError( cudaEventDestroy(startEvent), "kernel.cu", __LINE__  );
	cudaCheckAPIError( cudaEventDestroy(stopEvent), "kernel.cu", __LINE__  );
}

void nBodyHil()
{
	// Event parameters 
	cudaEvent_t startEvent, stopEvent;

	// Create the event using cudaEventCreate
	cudaCheckAPIError( cudaEventCreate(&startEvent), "kernel.cu", __LINE__  );
	cudaCheckAPIError( cudaEventCreate(&stopEvent), "kernel.cu", __LINE__  );

	int gridSize = N/ThreadsX;
	if(gridSize < 1)
		gridSize = 1;
	dim3 grid( gridSize );
	dim3 block(ThreadsX);

	cudaFuncSetCacheConfig(nbody_kernel_CODE, cudaFuncCachePreferL1);

	// compute body updates
	cudaCheckAPIError( cudaEventRecord(startEvent, 0), "kernel.cu", __LINE__  );
		nbody_kernel_CODE <<< grid, block >>> (b_out, index_d, b_in);
	results.push_back( completeEvent(startEvent, stopEvent) );

	// Release events
	cudaCheckAPIError( cudaEventDestroy(startEvent), "kernel.cu", __LINE__  );
	cudaCheckAPIError( cudaEventDestroy(stopEvent), "kernel.cu", __LINE__  );
}

void outputStats(std::vector<float>& results)
{
	// Median	
	std::sort( results.begin(), results.end());
	double med = 0.0;
	if(results.size()/2 == 0)
		med = results[ results.size()/2 ];
	else
	{
		med = (results[ results.size()/2 ] + results[ results.size()/2 - 1])/2.0; 
	}
	printf("Median: %.2f ms\n", med);

	results.clear();
}

void clearResults()
{
	results.clear();
	sresults.clear();
	vresults.clear();
	cresults.clear();
}

void sortBodies()
{
	// copy bodies from device memory into thrust device vector
	// first copy device to host,
	// fill in host vector with host buffer
	// copy from host to device vector
	// perform sort
	// copy device to host
	// update host buffer
	// copy to device memory
	// put this is render, then we have 1 less copy to host
	thrust::device_vector<body> b;
	thrust::host_vector<body> output;

	body bodies[N];

	// copy data from device to host
	//cudaCheckAPIError( cudaEventRecord(startEvent, 0) );
		cudaCheckAPIError( cudaMemcpy( bodies, b_in, sizeof(body)*N, cudaMemcpyDeviceToHost ), "kernel.cu", __LINE__  );

	for(int i = 0; i < N; i++)
	{
		output.push_back(bodies[i]);
		//bodies[i].print();
	}

	// Event parameters 
	cudaEvent_t startEvent, stopEvent;

	// Create the event using cudaEventCreate
	cudaCheckAPIError( cudaEventCreate(&startEvent), "kernel.cu", __LINE__  );
	cudaCheckAPIError( cudaEventCreate(&stopEvent), "kernel.cu", __LINE__  );

	b = output;

	// compute body updates
	cudaCheckAPIError( cudaEventRecord(startEvent, 0), "kernel.cu", __LINE__  );
		thrust::sort(b.begin(), b.end());
	sresults.push_back( completeEvent(startEvent, stopEvent) );

	// Release events
	cudaCheckAPIError( cudaEventDestroy(startEvent), "kernel.cu", __LINE__  );
	cudaCheckAPIError( cudaEventDestroy(stopEvent), "kernel.cu", __LINE__  );

	// cast thrust vector to raw pointer -- b needs to be global/error on close
	//b_in = thrust::raw_pointer_cast(b.data());

	output = b;
	
	for(int i= 0; i < N; i++)
	{
		bodies[i] = output[i];
	}

	cudaCheckAPIError( cudaMemcpy( b_out, bodies, sizeof(body)*N, cudaMemcpyHostToDevice), "kernel.cu", __LINE__  );

}

void sortRender(body* b_draw)
{
	body bodies[N];
	thrust::device_vector<body> b;
	thrust::host_vector<body> output;

	// read from device memory to host buffer
	cudaCheckAPIError( cudaMemcpy( bodies, b_draw, sizeof(body)*N, cudaMemcpyDeviceToHost ), "kernel.cu", __LINE__  );

	// set up drawing
    glEnable( GL_POINT_SMOOTH );
    glEnable( GL_BLEND );
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
    glPointSize( 9.0 );

	glClearColor( 0.0, 0.0, 1.0, 1.0 );
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// for all points - draw and copy into host vector
	glBegin(GL_POINTS);
		for(int i = 0; i < N; i++)
		{
			glColor3f( bodies[i].colour.x, bodies[i].colour.y, bodies[i].colour.z );	
			glVertex2f(bodies[i].position.x/(sim_rad/8), bodies[i].position.y/(sim_rad/8));
			output.push_back(bodies[i]);
		}
	glEnd();

	// finish drawing
	glFinish();
	glutSwapBuffers();

	// Event parameters 
	cudaEvent_t startEvent, stopEvent;

	// Create the event using cudaEventCreate
	cudaCheckAPIError( cudaEventCreate(&startEvent), "kernel.cu", __LINE__  );
	cudaCheckAPIError( cudaEventCreate(&stopEvent), "kernel.cu", __LINE__  );

	// copy from host to device vector
	b = output;

	// sort using thrust on morton code
	cudaCheckAPIError( cudaEventRecord(startEvent, 0), "kernel.cu", __LINE__  );
		thrust::sort(b.begin(), b.end());
	sresults.push_back( completeEvent(startEvent, stopEvent) );

	// Release events
	cudaCheckAPIError( cudaEventDestroy(startEvent), "kernel.cu", __LINE__  );
	cudaCheckAPIError( cudaEventDestroy(stopEvent), "kernel.cu", __LINE__  );

	// copy device to host vector
	output = b;
	
	// copy host vector to host memory
	for(int i= 0; i < N; i++)
	{
		bodies[i] = output[i];
		bodies[i].print();
	}
	printf("\n");

	//system("pause");
	// copy host memory to device memory
	cudaCheckAPIError( cudaMemcpy( b_in, bodies, sizeof(body)*N, cudaMemcpyHostToDevice), "kernel.cu", __LINE__  );
}

void outputBodies(body* b)
{
	body bodies[N];

	// copy data from device to host
	cudaCheckAPIError( cudaMemcpy( bodies, b, sizeof(body)*N, cudaMemcpyDeviceToHost ), "kernel.cu", __LINE__  );

	for(int i = 0; i < N; i++)
	{
		//output.push_back(bodies[i]);
		bodies[i].print();
	}
}

void Draw()
{	
	//printf("quad encoding..\n");

	executeQuadTree();

	//printf("sorting..\n");
	sortBodies();	
	
	// compute center of mass requires sorted order
	//printf("computing com..\n");
	computeCOM();

	//printf("performing interactions..\n");
	nBodyHil();

	//printf("drawing bodies..\n");
	renderBodies(b_out);

	//std::swap(b_in, b_out);
	if(benchmark)
	{
		static int i = 0;
		
		if(i == 0)
		{
			clearResults();
		}
		if(i > iterations)
		{
			i = 0;
		
			// Output Interaction Results
			printf("\nVoronoi NBody Results\n");
			printf("QuadBuild (%d)-\t", ThreadsQ);
			outputStats(vresults);
			printf("Sort -\t");
			outputStats(sresults);
			printf("CenterMass (%d) -\t", ThreadsC);
			outputStats(cresults);
			printf("NBody (%d) -\t", ThreadsX);
			outputStats(results);

			//ThreadsX *= 2;
			benchmark = false;
		}
		i++;
	}
}

void Draw1()
{
    glEnable( GL_POINT_SMOOTH );
    glEnable( GL_BLEND );
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
    glPointSize( 6.0 );

	glClearColor( 0.0, 0.0, 1.0, 1.0 );
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//glColor3f(1, 0, 0);

	glBegin(GL_POINTS);
		for(int i = 0; i < 1<<4*n; i++)
		{
			glColor3f( display_bodies[i].colour.x, display_bodies[i].colour.y, display_bodies[i].colour.z );	
			glVertex2f(display_bodies[i].position.x/(512), display_bodies[i].position.y/(512));
		}
	glEnd();

	glFinish();
	glutSwapBuffers();
}

void initGL(int argc, char *argv[], int wWidth, int wHeight)
{
	// init gl
	glutInit( &argc, argv );
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(500, 100);
	glutInitWindowSize( wWidth, wHeight );
	glutCreateWindow( "CUDA Nbody" );

	// set callback functions
	glutKeyboardFunc(Key);
	glutDisplayFunc(Draw);
    glutIdleFunc(Draw);
   
	// create another window
	glutInitWindowPosition(510+wWidth, 100);
	glutCreateWindow( "Buffer Area Usage" );
	glutDisplayFunc(Draw1);

   glewInit();
    if (glewIsSupported("GL_VERSION_2_1"))
        printf("Ready for OpenGL 2.1\n");
    else 
         printf("Warning: Detected that OpenGL 2.1 not supported\n");


	wglSwapIntervalEXT(true);
}

int2 findMYValue(int* LUT, int h)
{
	int DIM = 1 << n;

	for(int y = 0; y < DIM+2; y++)
	{
		for(int x = 0; x < DIM +2; x++)
		{
			if(LUT[y*(DIM+2) + x] == h)
			{
				int2 coords;
				coords.x = x - 1;
				coords.y = y - 1;
				return coords;
			}
		}
	}

	int2 err;
	err.x = -1;
	err.y = -1;
	return err;
}

void outputRegions(int2 coords, int* LUT)
{
	const int DIM = 1 << n;
	std::cout << LUT[ (coords.y) * (DIM+2) + coords.x     ] << " ";
	std::cout << LUT[ (coords.y) * (DIM+2) + coords.x + 1 ] << " ";
	std::cout << LUT[ (coords.y) * (DIM+2) + coords.x + 2 ] << " ";
	std::cout << std::endl;
	std::cout << LUT[ (coords.y + 1) * (DIM+2) + coords.x     ]<< " ";
	std::cout << LUT[ (coords.y + 1) * (DIM+2) + coords.x + 1 ]<< " ";
	std::cout << LUT[ (coords.y + 1) * (DIM+2) + coords.x + 2 ]<< " ";
	std::cout << std::endl;
	std::cout << LUT[ (coords.y + 2) * (DIM+2) + coords.x     ]<< " ";
	std::cout << LUT[ (coords.y + 2) * (DIM+2) + coords.x + 1 ]<< " ";
	std::cout << LUT[ (coords.y + 2) * (DIM+2) + coords.x + 2 ]<< " ";
	std::cout << std::endl;
}

int main(int argc, char** argv)
{
	try 
	{
		printf("N Body Benchmark CUDA\n\n");
		benchmark = true;

		// Initial body data
		const int DIM = 1 << n;

		printf("Number of Bodies : \t%d\n", N);
		std::vector<body> body_h;
		for(int i = 0; i < N; i++)
		{
			body_h.push_back( init_body(i) );
		}

		printf("Number of Zones - %d\n", 1<<2*n);
		printf("Simulation radius : %g\n", sim_rad);
		printf("Width of Region: %g\n", 2*sim_rad/DIM);

		// body_draw contains number of zones^2
		const int BUDIM = 1 << 2*n;
		int k = 0;
		double spacing_x = (2*512)/ BUDIM;
		double spacing_y = (2*512)/ BUDIM;
		display_bodies = (body*)malloc( sizeof(body)*(1 << 4*n) );		
		for(int y = -1*BUDIM/2; y < BUDIM/2; y++)
		{
			for(int x = -1*BUDIM/2; x < BUDIM/2; x++)
			{
				display_bodies[k].position.x = 1*spacing_x/2 + spacing_x*y;
				display_bodies[k].position.y = -1*spacing_y/2 - spacing_y*x;
				display_bodies[k].colour.x = 1.0f;
				display_bodies[k].colour.y = 0.0f;
				display_bodies[k].colour.z = 0.0f;
				display_bodies[k].colour.w = 1.0f;
				k++;
			}
		}
		printf("%d bodies created to show buffer usage\n\n", k);

		// Lookup Table for morton codes
		std::vector<int> padded;
		for(int y = 0; y < DIM+2; y++)
		{
			for(int x = 0; x < DIM+2; x++)
			{
				if(x == 0 || (x == DIM + 1) || y == 0 || (y == DIM + 1))
				{
					padded.push_back(-1);
				}
				else
				{
					padded.push_back( EncodeMorton2(x - 1, y - 1) );
				}
			}
		}

		// allocate memory on device for buffers
		cudaCheckAPIError( cudaMalloc( (void**)&b_in, sizeof(body)*body_h.size()), "kernel.cu", __LINE__  );
		cudaCheckAPIError( cudaMalloc( (void**)&b_out, sizeof(body)*body_h.size()), "kernel.cu", __LINE__  );
		cudaCheckAPIError( cudaMalloc( (void**)&LUT_d, sizeof(int)*(DIM+2)*(DIM+2)), "kernel.cu", __LINE__  );
		cudaCheckAPIError( cudaMalloc( (void**)&index_d, sizeof(int4)*(1<<2*n)), "kernel.cu", __LINE__  );
		cudaCheckAPIError( cudaMalloc( (void**)&b_draw,  sizeof(body)*(1 << 4*n) ), "kernel.cu", __LINE__  );

		// copy data from host to device
		cudaCheckAPIError( cudaMemcpy( b_in, &body_h[0], sizeof(body)*body_h.size(), cudaMemcpyHostToDevice), "kernel.cu", __LINE__  ); //same intial conditions
		cudaCheckAPIError( cudaMemcpy( LUT_d, &padded[0], sizeof(int)*(DIM+2)*(DIM+2), cudaMemcpyHostToDevice), "kernel.cu", __LINE__  );
		cudaCheckAPIError( cudaMemcpy( b_draw, display_bodies, sizeof(body)*(1 << 4*n), cudaMemcpyHostToDevice), "kernel.cu", __LINE__  ); //same intial conditions

		// run buffer Kernel
		bufferAreaKernel();

		// copy data from device to host
		cudaCheckAPIError( cudaMemcpy( display_bodies, b_draw, sizeof(body)*1<<4*n, cudaMemcpyDeviceToHost ), "kernel.cu", __LINE__  );

			initGL(argc, argv, 512, 512);
	}
	catch(thrust::system_error &err)
	{
		std::cerr << "Error : " << err.what() << std::endl;
		system("pause");
		cleanup();
		return EXIT_FAILURE;	
	}

        glutMainLoop();

	return 0;
}



// query device properties
void cudaQuery()
{
	// determine number of CUDA devices
	int count;
	cudaCheckAPIError( cudaGetDeviceCount(&count), "kernel.cu", __LINE__  );
	printLine("Number of CUDA Devices ", count);
	printBlank();

	// output information on all devices
	for(int i = 0; i < count; i++)
	{
		printLine("Device ", i+1);

		// determine properties
		cudaDeviceProp properties;
		cudaCheckAPIError( cudaGetDeviceProperties(&properties, i), "kernel.cu", __LINE__  );

		printLine("Name			", &properties.name);
		printLine("Total Global Mem	", properties.totalGlobalMem);
		printLine("Shared Mem Per Block	", properties.sharedMemPerBlock);
		printLine("Regs Per Block		", properties.regsPerBlock);
		printLine("Warp Size		", properties.warpSize);
		printLine("MemPitch		", properties.memPitch);
		printLine("Max Threads Per Block	", properties.maxThreadsPerBlock);
		printLine("Max Threads Dim		", properties.maxThreadsDim);
		printLine("Max Grid Size		", properties.maxGridSize);
		printLine("Total Const Mem		", properties.totalConstMem);
		printLine("Major			", properties.major);
		printLine("Minor			", properties.minor);
		printLine("Clock Rate		", properties.clockRate);
		printLine("Texture Alignment	", properties.textureAlignment);
		printLine("Device Overlap		", properties.deviceOverlap);
		printLine("Multi Processor Count	", properties.multiProcessorCount);
		printLine("Kernel Exec Timeout Enabled", properties.kernelExecTimeoutEnabled);
		printLine("Integrated		", properties.integrated);
		printLine("Can Map Host Memory	", properties.canMapHostMemory);
		printLine("Compute Mode		", properties.computeMode);
		printLine("Max Texture 1D		", properties.maxTexture1D);
		printLine("Max Surface 2D		", properties.maxSurface2D);
		printLine("Max Texture 2D		", properties.maxTexture2D);
		printLine("Max Texture 3D		", properties.maxTexture3D);
		printLine("Concurrent Kernels	", properties.concurrentKernels);
	}
	printBlank();
}

/*
__device__ int toDec(int* r, int b)
{
	int decimal = 0;
	int mult = 1;
    for(int i = 0; i < n; i++)
    {
        decimal += r[n-1-i]*mult;
		mult *= b;

    }

	return decimal;
}
__device__ int quadWord(int* code, int n)
{
	int word = 0;
	int mult = 1;
	for(int i = n-1; i > -1; i--)
	{
		word += code[i]*mult;
		mult *= 10;
	}
	return word;
}

__device__ int decToBase(int idx, int b)
{
    int r[n];
    for(int i = 0; i < n; i++)
    {
        r[n-1-i]=idx%b;
        idx /= b;
    }

	return quadWord(r, n);
}
*/
