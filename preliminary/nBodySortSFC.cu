/*
Voronoi N BODY

Voronoi Grid
NBody
Morton Coding
Hilbert Coding
Thrust Sort
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

const int N = 8;
const int uNumVoronoiPts = 16;
int ThreadsX = 4;
const int iterations = 100;
const double sim_rad = 1e18;

std::vector<float> results;
std::vector<float> vresults;
std::vector<float> sresults;

// Description of Voronoi Buf
struct VoronoiBuf
{
	double x;
	double y;
	int morton;
	int hilbert;
	float4 colour;
	double4 p;
	void print()
	{
		std::cout << "\tMorton: " << morton <<" Hilbert: " << hilbert << std::endl;
		std::cout <<"\tPosition(x,y): " << x << " " << y << std::endl;
		//std::cout << "\tR: " << colour.x << " G: " << colour.y << " B: " << colour.z << std::endl;
		std::cout << "\tCenter of Mass: " << p.x << " " << p.y << " - Mass: " << p.w << std::endl << std::endl;
	}
};

VoronoiBuf* Voronoi_d;

// Description of Body
__host__ __device__ struct body
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
/*
// sort Morton
__host__ __device__ bool operator<(const body &lhs, const body &rhs) 
{
	return lhs.morton < rhs.morton;
}
*/
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
	
	//if(output_thread)
		//printf("dist - %g\n", dist);
	
	// [6 FLOPS]
	double F = (G * bi.w * bj.w) / (dist*dist + EPS*EPS);
	//if(output_thread)
		//printf("F - %g\n", F);

	// [6 FLOPS]
	a.x += F * dx / dist;
	a.y += F * dy / dist;

	return a;
}

__global__ void nbody_kernel(body* body_in, body* body_out)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx >= N)
		return;

	extern __shared__ double4 shared_position[];

	double4 position = body_in[idx].position;
		
	// calculate force contributions for body
	double2 force;
	force.x = 0.0;
	force.y = 0.0;

	for(int tile = 0; tile < gridDim.x; tile++)
	{
		unsigned int k = tile * blockDim.x + threadIdx.x;
		shared_position[threadIdx.x] = body_in[k].position;

		__syncthreads();


//#pragma unroll 128
		for(unsigned int counter = 0; counter < blockDim.x; counter++)
		{
			force = bodyBodyInteraction(position, shared_position[counter], force, false);
		}

		__syncthreads();
	}
	
	// Do update
	double4 velocity = body_in[idx].velocity;

	double4 v;
	v.x = velocity.x + 1e10 * force.x / position.w;
	v.y = velocity.y + 1e10 * force.y / position.w;
	v.z = force.x;
	v.w = force.y;

	double4 p;
	p.x = position.x + 1e10 * v.x;
	p.y = position.y + 1e10 * v.y;
	p.z = 0;
	p.w = position.w;

	body_out[idx].velocity = v;
	body_out[idx].position = p;
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

	for(int i = 0; i < uNumVoronoiPts; i++)
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

	//center of mass for voronoi region
	double mass = body_in[idx].position.w;
	double2 com;
	com.x = v[minDistPoint].p.x * v[minDistPoint].p.w + x * mass;
	com.y = v[minDistPoint].p.y * v[minDistPoint].p.w + y * mass;
	double new_mass = mass + v[minDistPoint].p.w;

	v[minDistPoint].p.w = new_mass;
	v[minDistPoint].p.x = com.x/new_mass;
	v[minDistPoint].p.y = com.y/new_mass;

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
    glPointSize( 9.0 );

	glClearColor( 0.0, 0.0, 1.0, 1.0 );
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//glColor3f(1, 0, 0);

	glBegin(GL_POINTS);
		for(int i = 0; i < N; i++)
		{
			glColor3f( bodies[i].colour.x, bodies[i].colour.y, bodies[i].colour.z );	
			glVertex2f(bodies[i].position.x/(sim_rad/8), bodies[i].position.y/(sim_rad/8));
		}
	glEnd();

	glFinish();
	glutSwapBuffers();

	//for(int i = 0; i < N; i++)
	//	bodies[i].print();

	//system("pause");
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

void nBodySM()
{
	// Event parameters 
	cudaEvent_t startEvent, stopEvent;

	// Create the event using cudaEventCreate
	cudaCheckAPIError( cudaEventCreate(&startEvent) );
	cudaCheckAPIError( cudaEventCreate(&stopEvent) );

	dim3 grid(RoundUp(ThreadsX, N)/ThreadsX);
	dim3 block(ThreadsX);

	// compute body updates
	cudaCheckAPIError( cudaEventRecord(startEvent, 0) );
		nbody_kernel <<< grid, block, ThreadsX*sizeof(double4) >>> (b_out, b_in);
	results.push_back( completeEvent(startEvent, stopEvent) );

	// Release events
	cudaCheckAPIError( cudaEventDestroy(startEvent) );
	cudaCheckAPIError( cudaEventDestroy(stopEvent) );
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
	//printf("\t %.2f Mop/s\n", computeStats(med));
/*
	// Mean
	double sum = std::accumulate(std::begin(results), std::end(results), 0.0);
	double m =  sum / results.size();
	printf("Mean: %.2f ms\n", m);
	//printf("\t %.2f Mop/s\n", computeStats(m));

	// Standard deviation
	double accum = 0.0;
	std::for_each (std::begin(results), std::end(results), [&](const double d) {
		accum += (d - m) * (d - m);
	});
	double stdev = sqrt(accum / (results.size()-1));
	printf("Standard Deviation: %.2f\n", stdev);*/

	//printf("1. %.2f %d. %.2f\n", results[0], results.size()-1, results[results.size()-1]);
	results.clear();
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
		cudaCheckAPIError( cudaMemcpy( bodies, b_in, sizeof(body)*N, cudaMemcpyDeviceToHost ) );

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

void sortRender(body* b_draw)
{
	body bodies[N];
	thrust::device_vector<body> b;
	thrust::host_vector<body> output;

	// read from device memory to host buffer
	cudaCheckAPIError( cudaMemcpy( bodies, b_draw, sizeof(body)*N, cudaMemcpyDeviceToHost ) );

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
	cudaCheckAPIError( cudaEventCreate(&startEvent) );
	cudaCheckAPIError( cudaEventCreate(&stopEvent) );

	// copy from host to device vector
	b = output;

	// sort using thrust on morton code
	cudaCheckAPIError( cudaEventRecord(startEvent, 0) );
		thrust::sort(b.begin(), b.end());
	sresults.push_back( completeEvent(startEvent, stopEvent) );

	// Release events
	cudaCheckAPIError( cudaEventDestroy(startEvent) );
	cudaCheckAPIError( cudaEventDestroy(stopEvent) );

	// copy device to host vector
	output = b;
	
	// copy host vector to host memory
	for(int i= 0; i < N; i++)
	{
		bodies[i] = output[i];
		bodies[i].print();
	}
	printf("\n");
	// copy host memory to device memory
	cudaCheckAPIError( cudaMemcpy( b_in, bodies, sizeof(body)*N, cudaMemcpyHostToDevice) );
}

void Draw()
{	
	executeVoronoi();

	nBodySM();
	
	//sortBodies();
	//renderBodies(b_in);
	sortRender(b_in); // reduce memory transfers by 1

	static int i = 0;
	i++;
	if(i > iterations)
	{
		i = 0;
		
		// Output Interaction Results
		printf("\nVoronoi NBody Results\n");
		printf("Threads -\t%d\n", ThreadsX);
		printf("Voronoi -\t");
		outputStats(vresults);
		printf("NBody -\t");
		outputStats(results);
		printf("Sort -\t");
		outputStats(sresults);
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

	//wglSwapIntervalEXT(false);
}

int main(int argc, char** argv)
{
	printf("N Body Benchmark CUDA\n\n");

	// Initial body data
	const int body_size = sizeof(body)*N;
	const int voronoi_size = sizeof(VoronoiBuf)*uNumVoronoiPts;

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
	printf("Program Data\n");
	printf("Number of Voronoi Points :\t%d\n", uNumVoronoiPts);
	int k = 0;
	int dim = sqrt((float)uNumVoronoiPts);
	double spacing_x = (sim_rad/4)/ dim;
	double spacing_y = (sim_rad/4)/ dim;
	printf("%d %g %g\n\n", dim, spacing_x, spacing_y); 
	printf("Voronoi Points\n");
	for(int y = -1*dim/2; y < dim/2; y++)
	{
		for(int x = -1*dim/2; x < dim/2; x++)
		{
			Voronoi_h[k].x = spacing_x/2 + spacing_x*y;
			Voronoi_h[k].y = spacing_y/2 + spacing_y*x;
			Voronoi_h[k].colour.x = (sin(2.4*k + 0) *127 + 128)/255;
			Voronoi_h[k].colour.y = (sin(2.4*k + 2) *127 + 128)/255;
			Voronoi_h[k].colour.z = (sin(2.4*k + 4) *127 + 128)/255;
			Voronoi_h[k].colour.w = 1.0f;
			Voronoi_h[k].p.x = 0.0;
			Voronoi_h[k].p.y = 0.0;
			Voronoi_h[k].p.z = 0.0;
			Voronoi_h[k].p.w = 0.0;
			Voronoi_h[k].morton = EncodeMorton2(x + dim/2, y + dim/2);
			Voronoi_h[k].hilbert = EncodeHilbert2(uNumVoronoiPts, x + dim/2, y + dim/2);
			Voronoi_h[k].print();
			k++;
		}
	}

	// allocate memory on device for buffers
	cudaCheckAPIError( cudaMalloc( (void**)&b_in, body_size) );
	cudaCheckAPIError( cudaMalloc( (void**)&b_out, body_size) );
	cudaCheckAPIError( cudaMalloc( (void**)&Voronoi_d, voronoi_size) );


	// copy data from host to device
	cudaCheckAPIError( cudaMemcpy( b_in, body_h, body_size, cudaMemcpyHostToDevice) ); //same intial conditions
	cudaCheckAPIError( cudaMemcpy( Voronoi_d, Voronoi_h, voronoi_size, cudaMemcpyHostToDevice) );
	
	//free host memory
	free( Voronoi_h );
	free( body_h );
	
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
