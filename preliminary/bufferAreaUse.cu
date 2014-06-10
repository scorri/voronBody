/*
ADDED THIS TEXT
Buffer area usage
shows which regions other regions use in interactions

	0 1 2 3 4 5.....
   0
   1
   2
   3
   4
   .
   .

Morton Coding
http://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
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
#include <sstream>
#include <numeric>
#define _USE_MATH_DEFINES
#include "math.h"

#ifdef _WIN32
#include<windows.h>
#endif
#include <GL/glew.h>
#include <GL/wglew.h>
#include <GL/glut.h>

const int n = 2;
const int N = 1<<2*n;
int ThreadsX = 32;
const int sim_rad = 512;
int out;

std::vector<float> results;

void Hilbert();
void Morton();
void RM();
void TEST();

// Description of Body
struct body
{
	float4 colour;
	double2 position;
	bool operator==(body b)
	{
		return( (position.x == b.position.x) && (position.y == b.position.y) );
	}
	void print()
	{
		std::cout << "\n\tPosition(x,y): " << position.x << " " << position.y;		
	}
};

	int* LUT_d;
	body* b_draw;
	body* bodies;

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

__global__ void
reset_buffer_kernel(body* body_draw)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx > 1 << 4*n)
		return;

	float4 c;
	c.x = 1.0f;
	c.y = 0.0f;
	c.z = 0.0f;
	c.w = 1.0f;

	body_draw[idx].colour = c;
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

__global__ void 
buffer_area_kernel(body* body_draw, int* LUT)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx > N)
		return;

	int DIM = 1 << n;

	int2 coords = findValue(LUT, idx);

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
			body_draw[idx * N + regions[i]].colour = colour;
		}
	}
}

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

__global__ void 
BA_kernel_test(body* body_draw, int output)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx > 1<<2*n)
		return;

	int DIM = 1 << n;

	int2 coords;
	coords.x = DecodeMorton2X(idx);
	coords.y = DecodeMorton2Y(idx);

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
	if(idx == output)
	{
		for(int i = 0; i < 9; i++)
		{
			printf("%d ", regions[i]);
		}
		printf("\n");
	}

	if(idx == output)
	{
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
	cudaCheckAPIError( cudaFree( b_draw ) );
	cudaCheckAPIError( cudaFree( LUT_d ) );

	//free host memory
	free( bodies );

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
		case 'h':
			Hilbert();
			break;
		case 'm':
			Morton();
			break;
		case 'r':
			RM();
			break;
		case 't':
			TEST();
			break;
		case '+':
			out++;
			break;
		case '-':
			out--;
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

void renderBodies()
{
	std::stringstream ss;
	ss << out;
	std::string output;
	ss >> output;
	glutSetWindowTitle(output.c_str());

    glEnable( GL_POINT_SMOOTH );
    glEnable( GL_BLEND );
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
    glPointSize( 6.0 );

	glClearColor( 0.0, 0.0, 1.0, 1.0 );
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//glColor3f(1, 0, 0);

	glBegin(GL_POINTS);
		for(int i = 0; i < N*N; i++)
		{
			glColor3f( bodies[i].colour.x, bodies[i].colour.y, bodies[i].colour.z );	
			glVertex2f(bodies[i].position.x/(sim_rad), bodies[i].position.y/(sim_rad));
		}
	glEnd();

	glFinish();
	glutSwapBuffers();
}

void resetDisplayKernel()
{
	// Event parameters 
	cudaEvent_t startEvent, stopEvent;

	// Create the event using cudaEventCreate
	cudaCheckAPIError( cudaEventCreate(&startEvent) );
	cudaCheckAPIError( cudaEventCreate(&stopEvent) );

	int gridSize = (1 << 4*n)/ThreadsX;
	if(gridSize < 1)
		gridSize = 1;
	dim3 grid( gridSize );
	dim3 block(ThreadsX);

	cudaFuncSetCacheConfig(reset_buffer_kernel, cudaFuncCachePreferL1);

	// compute body updates
	cudaCheckAPIError( cudaEventRecord(startEvent, 0) );
		reset_buffer_kernel <<< grid, block >>> (b_draw );
	results.push_back( completeEvent(startEvent, stopEvent) );

	// Release events
	cudaCheckAPIError( cudaEventDestroy(startEvent) );
	cudaCheckAPIError( cudaEventDestroy(stopEvent) );
}

void bufferAreaKernel()
{
	// Event parameters 
	cudaEvent_t startEvent, stopEvent;

	// Create the event using cudaEventCreate
	cudaCheckAPIError( cudaEventCreate(&startEvent) );
	cudaCheckAPIError( cudaEventCreate(&stopEvent) );

	int gridSize = N/ThreadsX;
	if(gridSize < 1)
		gridSize = 1;
	dim3 grid( gridSize );
	dim3 block(ThreadsX);

	cudaFuncSetCacheConfig(buffer_area_kernel, cudaFuncCachePreferL1);

	// compute body updates
	cudaCheckAPIError( cudaEventRecord(startEvent, 0) );
		buffer_area_kernel <<< grid, block >>> (b_draw, LUT_d );
		printf("%.2f\n", completeEvent(startEvent, stopEvent) );
	//results.push_back( completeEvent(startEvent, stopEvent) );

	// Release events
	cudaCheckAPIError( cudaEventDestroy(startEvent) );
	cudaCheckAPIError( cudaEventDestroy(stopEvent) );
}

void Draw()
{	
	renderBodies();
}

void Draw1()
{

}

void initGL(int argc, char *argv[], int wWidth, int wHeight)
{
	// init gl
	glutInit( &argc, argv );
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(500, 100);
	glutInitWindowSize( wWidth, wHeight );

	// create window1
	glutCreateWindow( "Buffer Area Usage" );
	glutKeyboardFunc(Key);
	glutDisplayFunc(Draw);
    glutIdleFunc(Draw);
  /*
	// create another window
	glutInitWindowPosition(510+wWidth, 100);
	glutCreateWindow( "Other Window" );
	glutDisplayFunc(Draw1);
    glutIdleFunc(Draw1);
*/
   glewInit();
    if (glewIsSupported("GL_VERSION_2_1"))
        printf("Ready for OpenGL 2.1\n");
    else 
         printf("Warning: Detected that OpenGL 2.1 not supported\n");


	wglSwapIntervalEXT(false);
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

void Coding(const char* title, int* padded)
{
	const int DIM = 1 << n;

	glutSetWindowTitle(title);

	// reset display buffer
	resetDisplayKernel();

	// copy to device
	cudaCheckAPIError( cudaMemcpy( LUT_d, padded, sizeof(int)*(DIM+2)*(DIM+2), cudaMemcpyHostToDevice) );

	// run Kernel
	bufferAreaKernel();

	// copy data from device to host
	cudaCheckAPIError( cudaMemcpy( bodies, b_draw, sizeof(body)*N*N, cudaMemcpyDeviceToHost ) );

}

void TEST()
{
	const int DIM = 1 << n;

	glutSetWindowTitle("TEST");

	// reset display buffer
	resetDisplayKernel();

	// run Kernel
	// Event parameters 
	cudaEvent_t startEvent, stopEvent;

	// Create the event using cudaEventCreate
	cudaCheckAPIError( cudaEventCreate(&startEvent) );
	cudaCheckAPIError( cudaEventCreate(&stopEvent) );

	int gridSize = N/ThreadsX;
	if(gridSize < 1)
		gridSize = 1;
	dim3 grid( gridSize );
	dim3 block(ThreadsX);

	cudaFuncSetCacheConfig(BA_kernel, cudaFuncCachePreferL1);

	// compute body updates
	cudaCheckAPIError( cudaEventRecord(startEvent, 0) );
		BA_kernel <<< grid, block >>> (b_draw );
		printf("%.2f\n", completeEvent(startEvent, stopEvent) );
	//results.push_back( completeEvent(startEvent, stopEvent) );

	// Release events
	cudaCheckAPIError( cudaEventDestroy(startEvent) );
	cudaCheckAPIError( cudaEventDestroy(stopEvent) );

	// copy data from device to host
	cudaCheckAPIError( cudaMemcpy( bodies, b_draw, sizeof(body)*N*N, cudaMemcpyDeviceToHost ) );

}

void Morton()
{
	const int DIM = 1 << n;

	// create LUT
	printf("Morton LUT\n");
	int* padded = (int*)malloc(sizeof(int)*(DIM+2)*(DIM*2));
	int k = 0;
	for(int y = 0; y < DIM+2; y++)
	{
		for(int x = 0; x < DIM+2; x++)
		{
			if(x == 0 || (x == DIM + 1) || y == 0 || (y == DIM + 1))
			{
				padded[y*(DIM+2) + x] = -1;
			}
			else
			{
				padded[y*(DIM+2) + x] = EncodeMorton2(x-1, y-1);
				k++;
			}
			printf("%d\t", padded[y*(DIM+2) + x]);
		}
		printf("\n");
	}
	printf("\n");

	Coding("Morton coding Buffer Access", padded);

	//free host memory
	free( padded );
}

void Hilbert()
{
	const int DIM = 1 << n;

	// create LUT
	printf("Hilbert LUT\n");
	int* padded = (int*)malloc(sizeof(int)*(DIM+2)*(DIM*2));
	int k = 0;
	for(int y = 0; y < DIM+2; y++)
	{
		for(int x = 0; x < DIM+2; x++)
		{
			if(x == 0 || (x == DIM + 1) || y == 0 || (y == DIM + 1))
			{
				padded[y*(DIM+2) + x] = -1;
			}
			else
			{
				padded[y*(DIM+2) + x] = EncodeHilbert2((DIM)*(DIM), x-1, y-1);
				k++;
			}
			printf("%d\t", padded[y*(DIM+2) + x]);
		}
		printf("\n");
	}
	printf("\n");

	Coding("Hilbert coding Buffer Access", padded);

	//free host memory
	free( padded );
}

void RM()
{
	const int DIM = 1 << n;

	// create LUT
	printf("RM LUT\n");
	int* padded = (int*)malloc(sizeof(int)*(DIM+2)*(DIM*2));
	int k = 0;
	for(int y = 0; y < DIM+2; y++)
	{
		for(int x = 0; x < DIM+2; x++)
		{
			if(x == 0 || (x == DIM + 1) || y == 0 || (y == DIM + 1))
			{
				padded[y*(DIM+2) + x] = -1;
			}
			else
			{
				padded[y*(DIM+2) + x] = (x-1) + (y-1)*(DIM);
				k++;
			}
			printf("%d\t", padded[y*(DIM+2) + x]);
		}
		printf("\n");
	}
	printf("\n");

	Coding("Row-Major coding Buffer Access", padded);

	//free host memory
	free( padded );
}

int main(int argc, char** argv)
{
	// Initial body data
	const int DIM = 1 << n;
	out = 0;

	// body_draw contains number of zones^2
	printf("Display List\n");
	const int BUDIM = 1 << 2*n;
	int k = 0;
	double spacing_x = (2*sim_rad)/ BUDIM;
	double spacing_y = (2*sim_rad)/ BUDIM;
	bodies = (body*)malloc( sizeof(body)*(1 << 4*n) );		
	for(int y = -1*BUDIM/2; y < BUDIM/2; y++)
	{
		for(int x = -1*BUDIM/2; x < BUDIM/2; x++)
		{
			bodies[k].position.x = 1*spacing_x/2 + spacing_x*y;
			bodies[k].position.y = -1*spacing_y/2 - spacing_y*x;
			bodies[k].colour.x = 1.0f;
			bodies[k].colour.y = 0.0f;
			bodies[k].colour.z = 0.0f;
			bodies[k].colour.w = 1.0f;
			//printf("\t%d", k);
			bodies[k].print();
			k++;
		}
		//printf("\n");
	}
	printf("\n%d display bodies created\n", k);

	// allocate memory on device for buffers
	cudaCheckAPIError( cudaMalloc( (void**)&LUT_d, sizeof(int)*(DIM+2)*(DIM+2)) );
	cudaCheckAPIError( cudaMalloc( (void**)&b_draw,  sizeof(body)*(1 << 4*n) ) );

	// copy data from host to device
	cudaCheckAPIError( cudaMemcpy( b_draw, bodies, sizeof(body)*(1 << 4*n), cudaMemcpyHostToDevice) ); //same intial conditions

		initGL(argc, argv, 512, 512);
        glutMainLoop();

	return 0;
}
