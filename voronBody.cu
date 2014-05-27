/*
Voronoi N BODY

Voronoi Grid
NBody
Morton Coding
Hilbert Coding
Thrust Sort
Body Interactions for region
*/
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cmath>
#include <thrust/sort.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
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

const int N = 2048;
const int DIM = 1 << 4; // DIM = 2^n (n = 1, 2, 3..)
int ThreadsX = 128;
const int iterations = 100;
const double sim_rad = 1e18;

std::vector<float> results;
std::vector<float> vresults;
std::vector<float> sresults;
std::vector<float> cresults;

// Description of Voronoi Buf
struct VoronoiBuf
{
	double x;
	double y;
	int morton;
	int hilbert;
	float4 colour;
	void print()
	{
		std::cout << "\tMorton: " << morton <<" Hilbert: " << hilbert << std::endl;
		std::cout <<"\tPosition(x,y): " << x << " " << y << std::endl;
		//std::cout << "\tR: " << colour.x << " G: " << colour.y << " B: " << colour.z << std::endl;
	}
};

VoronoiBuf* Voronoi_d;
double4* M_cd;
int* LUT_d;
int2* index_d;

// Description of Body
struct body
{
	float4 colour;
	double4 position;
	double4 velocity;
	int morton;
	int hilbert;
	double2 force;
	bool operator==(body b)
	{
		return( (position.x == b.position.x) && (position.y == b.position.y) && (velocity.x == b.velocity.x) && (velocity.y == b.velocity.y) );
	}
	void error(body b)
	{
		double px = b.position.x - position.x;
		double py = b.position.y - position.y;
		double vx = b.velocity.x - velocity.x;
		double vy = b.velocity.y - velocity.y;
		double fx = b.velocity.z - velocity.z;
		double fy = b.velocity.w - velocity.w;

		std::cout << "Error in \n" << std::endl;
		std::cout << "\tP: " << px << " " << py;
		std::cout << std::endl;
		std::cout << "\tV: " << vx << " " << vy;
		std::cout << std::endl;
		std::cout << "\tF: " << fx << " " << fy;
		std::cout << std::endl;
	}
	void print()
	{
		std::cout << "\tMorton : " << morton << " Hilbert: " << hilbert;
		std::cout << "\n\tPosition(x,y): " << position.x << " " << position.y;
		std::cout << "\n\tV: " << velocity.x << " " << velocity.y;
		std::cout << "\tF: " << velocity.z << " " << velocity.w;
		std::cout << "\n\tMass: " << position.w << std::endl << std::endl;
	}
	void resetForce()
	{
		force = double2();
	}
	void addForce(body b)
	{
        double G = 6.67e-11;   // gravational constant
        double EPS = 3E4;      // softening parameter

        double dx = b.position.x - position.x;
        double dy = b.position.y - position.y;
        double dist = sqrt(dx*dx + dy*dy);
        double F = (G * position.w * b.position.w) / (dist*dist + EPS*EPS);
        force.x += F * dx / dist;
        force.y += F * dy / dist;
	}
	void update()
	{
        velocity.x += 1e10 * force.x / position.w;
        velocity.y += 1e10 * force.y / position.w;
        position.x += 1e10 * velocity.x;
        position.y += 1e10 * velocity.y;		
	}

};

// sort Hilbert
__host__ __device__ bool operator<(const body &lhs, const body &rhs) 
{
	return lhs.hilbert < rhs.hilbert;
}

// "Insert" a 0 bit after each of the 16 low bits of x
int Part1By1(int x)
{
	x &= 0x0000ffff;                  // x = ---- ---- ---- ---- fedc ba98 7654 3210
	x = (x ^ (x <<  8)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
	x = (x ^ (x <<  4)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
	x = (x ^ (x <<  2)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
	x = (x ^ (x <<  1)) & 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
	
	return x;
}

// "Insert" two 0 bits after each of the 10 low bits of x
int Part1By2(int x)
{
	x &= 0x000003ff;                  // x = ---- ---- ---- ---- ---- --98 7654 3210
	x = (x ^ (x << 16)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
	x = (x ^ (x <<  8)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
	x = (x ^ (x <<  4)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
	x = (x ^ (x <<  2)) & 0x09249249; // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
	
	return x;
}

int EncodeMorton2(int x, int y)
{
	return (Part1By1(y) << 1) + Part1By1(x);
}

int EncodeMorton3(int x, int y, int z)
{
	return (Part1By2(z) << 2) + (Part1By2(y) << 1) + Part1By2(x);
}

//rotate/flip a quadrant appropriately
void rot(int n, int *x, int *y, int rx, int ry) {
    if (ry == 0) {
        if (rx == 1) {
            *x = n-1 - *x;
            *y = n-1 - *y;
        }
 
        //Swap x and y
        int t  = *x;
        *x = *y;
        *y = t;
    }
}

//convert (x,y) to d
int EncodeHilbert2 (int n, int x, int y) {
    int rx, ry, s, d=0;
    for (s=n/2; s>0; s/=2) {
        rx = (x & s) > 0;
        ry = (y & s) > 0;
        d += s * s * ((3 * rx) ^ ry);
        rot(s, &x, &y, rx, ry);
    }
    return d;
}

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
		b.hilbert = 0;
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
		b.hilbert = 0;
		return b;
	}
}
//	thrust::device_vector<body> b;
	body* b_in;
	body* b_out;

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
	int left = 0;
	int right = N-1;

	while(right > left + 1)
	{
		int middle = (left + right)/2;
		if(body_in[middle].hilbert >= h)
			right = middle;
		else
			left = middle;
	}

	if(right < N-1 && body_in[right].hilbert == h)
		return right;

	return -1;
}

__global__ void
compute_center_mass_with_search(body* body_in, double4* c_m, int2* index_store)
{
    //	idx( 0 to uNumVoronoiPts )
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

	bool mass = 0;
	double4 center;
	center.x = 0.0;
	center.y = 0.0;
	center.z = 0.0;
	center.w = 0.0;

	int first = 0;
	int last = 0;

	int start = findFirstValue(body_in, idx);
	
	if(start != -1)
	{
		// for all bodies
		for(int b = start; b < N; b++)
		{
			int hil = body_in[b].hilbert;
			if(idx == hil)
			{
				// if first hilbert in list
				if(!mass)
				{
					center = body_in[b].position;	//assign initial value
					first = b;
					mass = true;
				}
				else
				{
					double4 other = body_in[b].position;
					center.x = center.x*center.w + other.x * other.w;
					center.y = center.y*center.w + other.y * other.w;
					//center.z = center.z*center.w + other.z * other.w;
					center.w += other.w;
					center.x = center.x/center.w;
					center.y = center.y/center.w;
					//center.z = center.z/center.w;				
				}
			}
			if(hil > idx)
			{
				first = (mass == 0) ? b : first;
				//if((!mass))
				//	first = b;
	
				last = b;
				break;
			}
		}
	}

	last = ((last == 0) && (first != 0)) ? N-1 : last;
	//if((last == 0) && (first != 0))
	//	last = N-1;

	//center of mass for voronoi region
	index_store[idx].x = first;
	index_store[idx].y = last;
	c_m[idx] = center;
}

__device__ double2 computeHilbertBodies(int h, int2* index_store, body* body_in, double4 position, double2 force)
{
	for(int i = index_store[h].x; i < index_store[h].y; i++)
	{
		force = bodyBodyInteraction(position, body_in[i].position, force, false);
	}

	return force;
}

__device__ int2 findValue(int* LUT, int h)
{
	for(int y = 0; y < DIM; y++)
	{
		for(int x = 0; x < DIM; x++)
		{
			if(LUT[y*DIM + x] == h)
			{
				int2 coords;
				coords.x = x;
				coords.y = y;
				return coords;
			}
		}
	}

	int2 err;
	err.x = -1;
	err.y = -1;
	return err;
}

__global__ void 
nbody_kernel_hil(body* body_in, double4* c_m, int* LUT, int2* index_store, body* body_out)
{
	// body[idx]
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

	// Indentify the regions neighbouring body[idx]
	int2 coords = findValue(LUT, body_in[idx].hilbert);
	int regions[9];

	// 9 reads from global memory.........
	regions[0] = LUT[(coords.y-1 + DIM)%DIM*DIM + (coords.x-1 + DIM)%DIM];
	regions[1] = LUT[(coords.y-1 + DIM)%DIM*DIM + (coords.x + DIM)%DIM];
	regions[2] = LUT[(coords.y-1 + DIM)%DIM*DIM + (coords.x+1 + DIM)%DIM];
		   
	regions[3] = LUT[(coords.y + DIM)%DIM*DIM + (coords.x-1 + DIM)%DIM];
	regions[4] = LUT[(coords.y + DIM)%DIM*DIM + (coords.x + DIM)%DIM];
	regions[5] = LUT[(coords.y + DIM)%DIM*DIM + (coords.x+1 + DIM)%DIM];
		   
	regions[6] = LUT[(coords.y+1 + DIM)%DIM*DIM + (coords.x-1 + DIM)%DIM];
	regions[7] = LUT[(coords.y+1 + DIM)%DIM*DIM + (coords.x + DIM)%DIM];
	regions[8] = LUT[(coords.y+1 + DIM)%DIM*DIM + (coords.x+1 + DIM)%DIM];

	// position of body[idx]
	double4 position = body_in[idx].position;

	// calculate force contributions for body
	double2 force;
	force.x = 0.0;
	force.y = 0.0;

	// f is force of centerofmass relations with neighbouring regions
	double2 f;
	f.x = 0.0;
	f.y = 0.0;
	for(int i = 0; i < 9; i++)
	{
		force = computeHilbertBodies(regions[i], index_store, body_in, position, force); 
		f = bodyBodyInteraction(position, c_m[regions[i]], f, false); 
	}

	// subtract the centerofmass influence to cancel when added to force
	force.x = force.x - f.x;
	force.y = force.y - f.y;

	// perform center of mass interactions with all regions
	for(int i = 0; i < DIM*DIM; i++)
	{
		double4 p_j = c_m[i];
		force = bodyBodyInteraction(position, p_j, force, false);
	}

	// Do update
	double4 velocity = body_in[idx].velocity;
	double2 v;
	v.x = velocity.x + 1e10 * force.x / position.w;
	v.y = velocity.y + 1e10 * force.y / position.w;

	body_out[idx].velocity.x = v.x;
	body_out[idx].velocity.y = v.y;
	body_out[idx].velocity.z = force.x;
	body_out[idx].velocity.w = force.y;

	body_out[idx].position.x = position.x + 1e10 * v.x;
	body_out[idx].position.y = position.y + 1e10 * v.y;
	body_out[idx].position.z = 0;
	body_out[idx].position.w = position.w;
	body_out[idx].colour = body_in[idx].colour;
	body_out[idx].morton = body_in[idx].morton;
	body_out[idx].hilbert = body_in[idx].hilbert;

}

// Create Voronoi kernel
__global__ void create_voronoi( body* body_in, body* body_out, VoronoiBuf * v)
{
    // map from thread to pixel position
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
	double x = body_in[idx].position.x;
	double y = body_in[idx].position.y;

	// if in image
	// set min dist to distance from 1st position in buffer
	double d_x = (v[0].x - x);
	double d_y = (v[0].y - y);
	double d = (d_x*d_x + d_y*d_y);

	double minDist = d;
	int minDistPoint = 0;

	for(int i = 0; i < DIM*DIM; i++)
	{
		double diff_x = (v[i].x - x);
		double diff_y = (v[i].y - y);
		double dist = (diff_x*diff_x + diff_y*diff_y);

		if(dist < minDist)
		{
			minDist = dist;
			minDistPoint = i;
		}
	}

	// now calculate the value at that position
	body_out[idx].morton = v[minDistPoint].morton;
	body_out[idx].hilbert = v[minDistPoint].hilbert;
	body_out[idx].colour = v[minDistPoint].colour;
	body_out[idx].position = body_in[idx].position;
	body_out[idx].velocity = body_in[idx].velocity;
}

bool cudaCheckAPIError(cudaError_t err)
{
	if(err != cudaSuccess)
	{
		std::cerr << "Error : " << cudaGetErrorString(err) << std::endl;
		system("pause");
		return false;
	}

	return true;
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
	cudaCheckAPIError( cudaFree( b_in ) );
	cudaCheckAPIError( cudaFree( b_out ) );
	cudaCheckAPIError( cudaFree( M_cd) );
	cudaCheckAPIError( cudaFree( LUT_d ) );
	cudaCheckAPIError( cudaFree( Voronoi_d ) );

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
		case 'p':
			system("pause");
			break;
    }
}

float completeEvent(cudaEvent_t start, cudaEvent_t stop)
{
	// Add the stop event to the GPUs queue of work
	cudaCheckAPIError( cudaEventRecord(stop, 0) );
	
	// Wait until the event has completed so it is safe to read
	cudaCheckAPIError( cudaEventSynchronize(stop) );
	
	// Determine the time elapsed between the events
	float milliseconds = 0;
	cudaCheckAPIError( cudaEventElapsedTime(&milliseconds, start, stop) );

	return milliseconds;
}

void renderBodies(body* b)
{
	body bodies[N];

	// copy data from device to host
	//cudaCheckAPIError( cudaEventRecord(startEvent, 0) );
		cudaCheckAPIError( cudaMemcpy( bodies, b, sizeof(body)*N, cudaMemcpyDeviceToHost ) );
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
	double4 com[DIM*DIM];
	int2 index_store[DIM*DIM];

	// copy data from device to host
	cudaCheckAPIError( cudaMemcpy( com, M_cd, sizeof(double4)*DIM*DIM, cudaMemcpyDeviceToHost ) );
	cudaCheckAPIError( cudaMemcpy( index_store, index_d, sizeof(int2)*DIM*DIM, cudaMemcpyDeviceToHost ) );

	for(int i = 0; i < DIM*DIM; i++)
	{
		std::cout <<"H" << i << "\tX: " << com[i].x << " Y: " << com[i].y << " M: " << com[i].w << std::endl;
		std::cout <<"\tFirst" << index_store[i].x << " Last: " << index_store[i].y << std::endl;
	}
	system("pause");
}

// Execute voronoi kernel
void executeVoronoi()
{
	// Event parameters 
	cudaEvent_t startEvent, stopEvent;

	// Create the event using cudaEventCreate
	cudaCheckAPIError( cudaEventCreate(&startEvent) );
	cudaCheckAPIError( cudaEventCreate(&stopEvent) );

	dim3 grid( N/ThreadsX );
	dim3 block( ThreadsX );

	cudaFuncSetCacheConfig(create_voronoi, cudaFuncCachePreferL1);

	cudaCheckAPIError( cudaEventRecord(startEvent, 0) );
		create_voronoi <<< grid, block >>> (b_in, b_out, Voronoi_d);
	vresults.push_back( completeEvent(startEvent, stopEvent) );

	// Release events
	cudaCheckAPIError( cudaEventDestroy(startEvent) );
	cudaCheckAPIError( cudaEventDestroy(stopEvent) );
}

// compute center of mass
void computeCOM()
{
	// Event parameters 
	cudaEvent_t startEvent, stopEvent;

	// Create the event using cudaEventCreate
	cudaCheckAPIError( cudaEventCreate(&startEvent) );
	cudaCheckAPIError( cudaEventCreate(&stopEvent) );

	dim3 grid( 1 );
	dim3 block( DIM*DIM );

	cudaFuncSetCacheConfig(compute_center_mass_with_search, cudaFuncCachePreferL1);

	cudaCheckAPIError( cudaEventRecord(startEvent, 0) );
		compute_center_mass_with_search <<< grid, block >>> (b_in, M_cd, index_d);
	cresults.push_back( completeEvent(startEvent, stopEvent) );

	// Release events
	cudaCheckAPIError( cudaEventDestroy(startEvent) );
	cudaCheckAPIError( cudaEventDestroy(stopEvent) );
}

void nBodyHil()
{
	// Event parameters 
	cudaEvent_t startEvent, stopEvent;

	// Create the event using cudaEventCreate
	cudaCheckAPIError( cudaEventCreate(&startEvent) );
	cudaCheckAPIError( cudaEventCreate(&stopEvent) );

	dim3 grid(RoundUp(ThreadsX, N)/ThreadsX);
	dim3 block(ThreadsX);

	//printf("%d %d\n", grid.x, block.x);
	//system("pause");

	cudaFuncSetCacheConfig(nbody_kernel_hil, cudaFuncCachePreferL1);

	// compute body updates
	cudaCheckAPIError( cudaEventRecord(startEvent, 0) );
		nbody_kernel_hil <<< grid, block >>> (b_in, M_cd, LUT_d, index_d, b_out);
	results.push_back( completeEvent(startEvent, stopEvent) );

	// Release events
	cudaCheckAPIError( cudaEventDestroy(startEvent) );
	cudaCheckAPIError( cudaEventDestroy(stopEvent) );
}

float outputStats(std::vector<float>& results)
{
	// Median	
	std::sort( results.begin(), results.end());
	float med = 0.0f;
	if(results.size()/2 == 0)
		med = results[ results.size()/2 ];
	else
	{
		med = (results[ results.size()/2 ] + results[ results.size()/2 - 1])/2.0; 
	}
	printf("Median: %.2f ms\n", med);
	//printf("\t %.2f Mop/s\n", computeStats(med));

	//printf("1. %.2f %d. %.2f\n", results[0], results.size()-1, results[results.size()-1]);
	results.clear();

	return med;
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
		cudaCheckAPIError( cudaMemcpy( bodies, b_out, sizeof(body)*N, cudaMemcpyDeviceToHost ) );

	for(int i = 0; i < N; i++)
	{
		output.push_back(bodies[i]);
		//bodies[i].print();
	}

	// Event parameters 
	cudaEvent_t startEvent, stopEvent;

	// Create the event using cudaEventCreate
	cudaCheckAPIError( cudaEventCreate(&startEvent) );
	cudaCheckAPIError( cudaEventCreate(&stopEvent) );

	b = output;

	// compute body updates
	cudaCheckAPIError( cudaEventRecord(startEvent, 0) );
		thrust::sort(b.begin(), b.end());
	sresults.push_back( completeEvent(startEvent, stopEvent) );

	// Release events
	cudaCheckAPIError( cudaEventDestroy(startEvent) );
	cudaCheckAPIError( cudaEventDestroy(stopEvent) );

	// cast thrust vector to raw pointer -- b needs to be global/error on close
	//b_in = thrust::raw_pointer_cast(b.data());

	output = b;
	
	for(int i= 0; i < N; i++)
	{
		bodies[i] = output[i];
	}

	cudaCheckAPIError( cudaMemcpy( b_in, bodies, sizeof(body)*N, cudaMemcpyHostToDevice) );
	
/*
	printf("\nSorted\n");
	for(unsigned int i = 0; i < output.size(); i++)
		output[i].print();

	system("pause");	
*/
}

void Draw()
{	
	executeVoronoi();

	sortBodies();	

	// compute center of mass requires sorted order
	computeCOM();
	//outputCOM();

	//nBodySM();
	//nBodyOrig();
	nBodyHil();

	renderBodies(b_out);

	std::swap(b_in, b_out);

	static int i = 0;
	i++;
	if(i > iterations)
	{
		i = 0;
		
		float total_time = 0.0f;
		// Output Interaction Results
		printf("\nVoronoi NBody Results\n");
		printf("Threads -\t%d\n", ThreadsX);
		printf("Voronoi -\t");
		total_time += outputStats(vresults);
		printf("Sort -\t");
		total_time += outputStats(sresults);
		printf("CoM -\t");
		total_time += outputStats(cresults);
		printf("NBody -\t");
		total_time += outputStats(results);
		printf("Total Time for simulation - %.2f\n", total_time);
		ThreadsX *= 2;

	//	system("pause");
	}

	if((ThreadsX > N) || (ThreadsX > 1024))
	{
		system("pause");
		cleanup();
	}

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
   
   glewInit();
    if (glewIsSupported("GL_VERSION_2_1"))
        printf("Ready for OpenGL 2.1\n");
    else 
         printf("Warning: Detected that OpenGL 2.1 not supported\n");


	wglSwapIntervalEXT(false);
}
/*
int2 findValue(int** LUT, int h)
{
	int n = 4;

	for(int y = 0; y < n; y++)
	{
		for(int x = 0; x < n; x++)
		{
			if(LUT[x][y] == h)
			{
				int2 coords;
				coords.x = x;
				coords.y = y;
				return coords;
			}
		}
	}

	int2 err;
	err.x = -1;
	err.y = -1;
	return err;
}

int2 findValue(int* LUT, int h)
{
	int n = 4;

	for(int y = 0; y < n; y++)
	{
		for(int x = 0; x < n; x++)
		{
			if(LUT[y*n + x] == h)
			{
				int2 coords;
				coords.x = x;
				coords.y = y;
				return coords;
			}
		}
	}

	int2 err;
	err.x = -1;
	err.y = -1;
	return err;
}

void outputSearchZone(int** LUT, int zone)
{
	int n = 4;
	//find index of LUT for Hilbert zone subject of search
	int2 idx = findValue(LUT, zone);

	std::cout << "\nSearch zone " << zone << " is at index - " << idx.x << ", "<< idx.y << std::endl;
	//std::cout << (idx - n - 1 + uNumVoronoiPts)%uNumVoronoiPts << " " << (idx - n + uNumVoronoiPts)%uNumVoronoiPts << " "<< (idx - n + 1 + uNumVoronoiPts)%uNumVoronoiPts << std::endl;
	//std::cout << (idx - 1 + uNumVoronoiPts)%uNumVoronoiPts << " " << idx << " "<< (idx + 1 + uNumVoronoiPts)%uNumVoronoiPts << std::endl;
	//std::cout << (idx + n - 1 + uNumVoronoiPts)%uNumVoronoiPts << " " << (idx + n + uNumVoronoiPts)%uNumVoronoiPts << " "<< (idx + n + 1 + uNumVoronoiPts)%uNumVoronoiPts << std::endl;
	std::cout << LUT[(idx.x-1 + n)%n][(idx.y-1 + n)%n] << " " << LUT[(idx.x + n)%n][(idx.y-1 + n)%n]<< " "<< LUT[(idx.x+1 + n)%n][(idx.y-1 + n)%n] << std::endl;
	std::cout << LUT[(idx.x-1 + n)%n][(idx.y + n)%n] << " " << LUT[(idx.x + n)%n][(idx.y + n)%n] << " "<< LUT[(idx.x+1 + n)%n][(idx.y + n)%n] << std::endl;
	std::cout << LUT[(idx.x-1 + n)%n][(idx.y+1 + n)%n] << " " << LUT[(idx.x + n)%n][(idx.y+1 + n)%n] << " "<< LUT[(idx.x+1 + n)%n][(idx.y+1 + n)%n] << std::endl;
}

void outputSearchZone(int* LUT, int zone)
{
	int n = 4;
	//find index of LUT for Hilbert zone subject of search
	int2 coords = findValue(LUT, zone);

	std::cout << "\nSearch zone " << zone << " is at index - " << coords.x << ", "<< coords.y << std::endl;
	//std::cout << (idx - n - 1 + uNumVoronoiPts)%uNumVoronoiPts << " " << (idx - n + uNumVoronoiPts)%uNumVoronoiPts << " "<< (idx - n + 1 + uNumVoronoiPts)%uNumVoronoiPts << std::endl;
	//std::cout << (idx - 1 + uNumVoronoiPts)%uNumVoronoiPts << " " << idx << " "<< (idx + 1 + uNumVoronoiPts)%uNumVoronoiPts << std::endl;
	//std::cout << (idx + n - 1 + uNumVoronoiPts)%uNumVoronoiPts << " " << (idx + n + uNumVoronoiPts)%uNumVoronoiPts << " "<< (idx + n + 1 + uNumVoronoiPts)%uNumVoronoiPts << std::endl;
	std::cout << LUT[(coords.y-1 + n)%n*n + (coords.x-1 + n)%n] 
	<< " " << LUT[(coords.y-1 + n)%n*n + (coords.x + n)%n]<< " "
		<< LUT[(coords.y-1 + n)%n*n + (coords.x+1 + n)%n] << std::endl;
	
	std::cout << LUT[(coords.y + n)%n*n + (coords.x-1 + n)%n] << " " 
		<< LUT[(coords.y + n)%n*n + (coords.x + n)%n] << " "
		<< LUT[(coords.y + n)%n*n + (coords.x+1 + n)%n]<< std::endl;
	
	std::cout << LUT[(coords.y+1 + n)%n*n + (coords.x-1 + n)%n] << " " 
		<< LUT[(coords.y+1 + n)%n*n + (coords.x + n)%n] << " "
		<< LUT[(coords.y+1 + n)%n*n + (coords.x+1 + n)%n]<< std::endl;
}
*/
int main(int argc, char** argv)
{
	printf("N Body Benchmark CUDA\n\n");

	// Initial body data
	const int body_size = sizeof(body)*N;
	const int voronoi_size = sizeof(VoronoiBuf)*DIM*DIM;

	printf("Body List\n");
	body* body_h = (body*)malloc( body_size );
	for(int i = 0; i < N; i++)
	{
		body_h[i] = init_body(i);
		//body_h[i].print();
	}
	printf("\n");

	// Generate Voronoi Points	
	VoronoiBuf* Voronoi_h = (VoronoiBuf*)malloc(voronoi_size);
	int** LUT_h = (int**)malloc(sizeof(int*)*DIM);
	for(int i = 0; i < DIM; i++)
		LUT_h[i] = (int*)malloc(sizeof(int)*DIM);
	int* lut_h = (int*)malloc(sizeof(int)*DIM*DIM);

	printf("Program Data\n");
	printf("Number of Voronoi Points :\t%d\n", DIM*DIM);
	int k = 0;
	
	double radius = sim_rad/8;
	double spacing_x = (2*radius)/ DIM;
	double spacing_y = (2*radius)/ DIM;
	printf("%d %g %g\n\n", DIM, spacing_x, spacing_y); 
	printf("Voronoi Points\n");
	for(int y = -1*DIM/2; y < DIM/2; y++)
	{
		for(int x = -1*DIM/2; x < DIM/2; x++)
		{
			Voronoi_h[k].x = spacing_x/2 + spacing_x*y;
			Voronoi_h[k].y = spacing_y/2 + spacing_y*x;
			Voronoi_h[k].colour.x = (sin(2.4*k + 0) *127 + 128)/255;
			Voronoi_h[k].colour.y = (sin(2.4*k + 2) *127 + 128)/255;
			Voronoi_h[k].colour.z = (sin(2.4*k + 4) *127 + 128)/255;
			Voronoi_h[k].colour.w = 1.0f;
			Voronoi_h[k].morton = EncodeMorton2(x + DIM/2, y + DIM/2);
			Voronoi_h[k].hilbert = EncodeHilbert2(DIM*DIM, x + DIM/2, y + DIM/2);
			//Voronoi_h[k].print();
			LUT_h[x+DIM/2][y+DIM/2] = Voronoi_h[k].hilbert;
			lut_h[k] = Voronoi_h[k].hilbert;
			//printf("%d - %d\t", k, LUT_h[x+DIM/2][y+DIM/2]);
			k++;
		}
		//std::cout << std::endl;
	}
	/*
	double spacing_x = (sim_rad/4)/ DIM;
	double spacing_y = (sim_rad/4)/ DIM;
	printf("%d %g %g\n\n", DIM, spacing_x, spacing_y); 
	printf("Voronoi Points\n");
	for(int y = -1*DIM/2; y < DIM/2; y++)
	{
		for(int x = -1*DIM/2; x < DIM/2; x++)
		{
			Voronoi_h[k].x = spacing_x/2 + spacing_x*y;
			Voronoi_h[k].y = spacing_y/2 + spacing_y*x;
			Voronoi_h[k].colour.x = (sin(2.4*k + 0) *127 + 128)/255;
			Voronoi_h[k].colour.y = (sin(2.4*k + 2) *127 + 128)/255;
			Voronoi_h[k].colour.z = (sin(2.4*k + 4) *127 + 128)/255;
			Voronoi_h[k].colour.w = 1.0f;
			Voronoi_h[k].morton = EncodeMorton2(x + DIM/2, y + DIM/2);
			Voronoi_h[k].hilbert = EncodeHilbert2(DIM*DIM, x + DIM/2, y + DIM/2);
			Voronoi_h[k].print();
			LUT_h[x+DIM/2][y+DIM/2] = Voronoi_h[k].hilbert;
			lut_h[k] = Voronoi_h[k].hilbert;
			printf("%d - %d\t", k, LUT_h[x+DIM/2][y+DIM/2]);
			k++;
		}
		std::cout << std::endl;
	}*/
	/*
	for(int i = 0; i < uNumVoronoiPts; i++)
	{
		outputSearchZone(LUT_h, i);
		std::cout<<std::endl;
		outputSearchZone(lut_h, i);
	}
	*/

	// allocate memory on device for buffers
	cudaCheckAPIError( cudaMalloc( (void**)&b_in, body_size) );
	cudaCheckAPIError( cudaMalloc( (void**)&b_out, body_size) );
	cudaCheckAPIError( cudaMalloc( (void**)&Voronoi_d, voronoi_size) );
	cudaCheckAPIError( cudaMalloc( (void**)&LUT_d, sizeof(int)*DIM*DIM) );
	cudaCheckAPIError( cudaMalloc( (void**)&M_cd, sizeof(double4)*DIM*DIM) );
	cudaCheckAPIError( cudaMalloc( (void**)&index_d, sizeof(int2)*DIM*DIM) );

	// copy data from host to device
	cudaCheckAPIError( cudaMemcpy( b_in, body_h, body_size, cudaMemcpyHostToDevice) ); //same intial conditions
	cudaCheckAPIError( cudaMemcpy( Voronoi_d, Voronoi_h, voronoi_size, cudaMemcpyHostToDevice) );
	cudaCheckAPIError( cudaMemcpy( LUT_d, lut_h, sizeof(int)*DIM*DIM, cudaMemcpyHostToDevice) );
	
	//free host memory
	free( Voronoi_h );
	free( body_h );
	for(int i = 0; i < DIM; i++)
		free(LUT_h[i]);
	free( LUT_h );
	free( lut_h );
	
	// Output some useful data
	printf("Number of Bodies : \t%d\n", N);

	printf("NBody\n");
	printf("Global Work Size :\t%d\n", RoundUp(ThreadsX, N)/ThreadsX );
	printf("Local Work Size :\t%d\n\n\n", ThreadsX);

		initGL(argc, argv, 512, 512);
        glutMainLoop();

	return 0;
}



// query device properties
void cudaQuery()
{
	// determine number of CUDA devices
	int count;
	cudaCheckAPIError( cudaGetDeviceCount(&count) );
	printLine("Number of CUDA Devices ", count);
	printBlank();

	// output information on all devices
	for(int i = 0; i < count; i++)
	{
		printLine("Device ", i+1);

		// determine properties
		cudaDeviceProp properties;
		cudaCheckAPIError( cudaGetDeviceProperties(&properties, i) );

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
