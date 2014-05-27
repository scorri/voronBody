/*
Voronoi Example
Uses original methodology from the HandsOnLabs
example for DirectCompute Samples.
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <algorithm>
#include <vector>
#include <numeric>
#include "FreeImage.h"
#include "GL/glut.h"

// Simulation parameters
const int width = 512;
const int height = 512;
const int ThreadsX = 16;
const int ThreadsY = 16;
const int uNumVoronoiPts = 16;
const int iterations = 100;

// Description of Voronoi Buf
struct VoronoiBuf
{
	int x;
	int y;
	int r;
	int g;
	int b;
};

// Desciption of Profile
// used to store generated results
struct profile
{
	int threadsX;
	int threadsY;
	double median;
	double mean;
	double stddev;
	double perf;
	profile(int _tx, int _ty, double _med, double _mean, double sd)
	{
		threadsX = _tx; threadsY = _ty; median = _med, mean = _mean; stddev = sd; perf = 0.0;
	}
	void print()
	{
		perf = uNumVoronoiPts * width * height * 1e-6 / median;
		printf("\nVoronoi Results\n");
		printf("\tThreadsX: %d\n", threadsX);
		printf("\tThreadsY: %d\n", threadsY);
		printf("\tMedian: %.2f ms\n", median);	
		printf("\tMean: %.2f ms\n", mean);
		printf("\tStandard Deviation: %.2f\n", stddev);
		printf("\tInteractions per second: %.2f\n", perf);
		printf("\tGFlops: %.2f\n", perf * 28.5);
	}
};

// Performance stats
std::vector<float> results;
std::vector<profile> records;

// Globals for graphics
unsigned char* image_data;
unsigned char* output;
VoronoiBuf* Voronoi_d;

// Output CUDA device information
void cudaQuery();

// Create Voronoi kernel
__global__ void create_voronoi( unsigned char* image_data, VoronoiBuf * v)
{
    // map from thread to pixel position
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int offset = y * width + x;

	// if in image
	if(x < width && y < height)
	{
		int minDist = 99999;
		int minDistPoint = 0;

		for(int i = 0; i < uNumVoronoiPts; i++)
		{
			int diff_x = (v[i].x - x);
			int diff_y = (v[i].y - y);
			int dist = (diff_x*diff_x + diff_y*diff_y);

			if(dist < minDist)
			{
				minDist = dist;
				minDistPoint = i;
			}
		}

		if(minDist < 25)
		{
			// now calculate the value at that position
			image_data[offset*4 + 0] = v[minDistPoint].r/2;
			image_data[offset*4 + 1] = v[minDistPoint].g/2;
			image_data[offset*4 + 2] = v[minDistPoint].b/2;
			image_data[offset*4 + 3] = 255;
		}
		else
		{
			// now calculate the value at that position
			image_data[offset*4 + 0] = v[minDistPoint].r;
			image_data[offset*4 + 1] = v[minDistPoint].g;
			image_data[offset*4 + 2] = v[minDistPoint].b;
			image_data[offset*4 + 3] = 255;
		}
	}
}

// Save image using FreeImage library
bool saveImage(std::string file, unsigned char* in_buffer)
{
    FREE_IMAGE_FORMAT format = FreeImage_GetFIFFromFilename(file.c_str());
    FIBITMAP *image = FreeImage_ConvertFromRawBits((BYTE*)in_buffer, width,
                        height, width * 4, 32,
                        0xFF000000, 0x00FF0000, 0x0000FF00);
	if(!FreeImage_Save(format, image, file.c_str()))
		return false;

	return true;
}

// Check for a Cuda Error and output error info
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

// for printing information
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

// Round up to nearest multiple
size_t roundUp(int groupSize, int globalSize)
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

// Cleanup
void cleanup()
{
	// Free host memory
	free( output );

	// Free device memory
	cudaCheckAPIError( cudaFree( Voronoi_d ) );
	cudaCheckAPIError( cudaFree( image_data ) );

	system("pause");

	// Exit Application
	exit(EXIT_SUCCESS);
}

// Check keyboard inputs
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

// Main render call
void Draw()
{
	glClearColor( 0.0, 0.0, 0.0, 1.0 );
	glClear( GL_COLOR_BUFFER_BIT );
	glDrawPixels( width, height, GL_RGBA, GL_UNSIGNED_BYTE, output );
	glFlush();
}

// Wait for event to complete and return timing info
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

// Read results buffer and save to file
void saveVoronoi(const char* file)
{
	// Allocate host memory
	unsigned char* output;
	output = (unsigned char*)malloc(width*height*4);

	// Transfer results from device to host
	cudaCheckAPIError( cudaMemcpy(output, image_data, sizeof(int)*width*height, cudaMemcpyDeviceToHost) );

	// use free image to save output image to check correctness
	if( saveImage(file, output) )
		printf("Image saved\n\n");

	// free memory
	free(output);
}

// Calculate the median, mean and standard deviation of recorded timings
// saves to records and clears results
void recordStats(std::vector<float>& results)
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

	// Mean
	double sum = std::accumulate(std::begin(results), std::end(results), 0.0);
	double m =  sum / results.size();

	// Standard deviation
	double accum = 0.0;
	std::for_each (std::begin(results), std::end(results), [&](const double d) {
		accum += (d - m) * (d - m);
	});
	double stdev = sqrt(accum / (results.size()-1));

	// record stats
	records.push_back(profile(ThreadsX, ThreadsY, med, m, stdev) );

	// clear results
	results.clear();
}

// Execute voronoi kernel
void executeVoronoiBM()
{
	// Event parameters 
	cudaEvent_t startEvent, stopEvent;

	// Create the event using cudaEventCreate
	cudaCheckAPIError( cudaEventCreate(&startEvent) );
	cudaCheckAPIError( cudaEventCreate(&stopEvent) );

	dim3 grid( width/ThreadsX, height/ThreadsY);
	dim3 block( ThreadsX, ThreadsY );

	cudaFuncSetCacheConfig(create_voronoi, cudaFuncCachePreferL1);

	for(int i = 0; i < iterations; i++)
	{
		cudaCheckAPIError( cudaEventRecord(startEvent, 0) );
			create_voronoi <<< grid, block >>> (image_data, Voronoi_d);
		results.push_back( completeEvent(startEvent, stopEvent) );
	}

	recordStats(results);
	records[0].print();

	// Release events
	cudaCheckAPIError( cudaEventDestroy(startEvent) );
	cudaCheckAPIError( cudaEventDestroy(stopEvent) );
}

int main(int argc, char** argv)
{
	printf("Voronoi Benchmark CUDA\n\n");

	const int image_size = width * height * 4;	
	const int voronoi_size = roundUp(16, sizeof(VoronoiBuf)*uNumVoronoiPts);

	VoronoiBuf* Voronoi_h = (VoronoiBuf*)malloc(voronoi_size);
	//output = (unsigned char*)malloc(image_size);
	
	cudaCheckAPIError( cudaMalloc( (void**)&image_data, image_size) );
	cudaCheckAPIError( cudaMalloc( (void**)&Voronoi_d, voronoi_size) );

	// Generate Voronoi Points
	printf("Program Data\n");
	printf("Number of Voronoi Points :\t%d\n", uNumVoronoiPts);
	int k = 0;
	int dim = sqrt((float)uNumVoronoiPts);
	int spacing_x = width/ dim;
	int spacing_y = height/ dim;
	printf("%d %d %d\n", dim, spacing_x, spacing_y); 
	for(int i = 0; i < dim; i++)
	{
		for(int j = 0; j < dim; j++)
		{
			Voronoi_h[k].x = spacing_x/2 + spacing_x*i;
			Voronoi_h[k].y = spacing_y/2 + spacing_y*j;
			Voronoi_h[k].r = 25 + 204 * (rand()%256)/255;
			Voronoi_h[k].g = 25 + 204 * (rand()%256)/255;
			Voronoi_h[k].b = 25 + 204 * (rand()%256)/255;	
			printf("%d %d\n", Voronoi_h[k].x, Voronoi_h[k].y);
			k++;
		}
	}
	
	printf("Image size :\t%d %d\n", width, height);
	printf("Grid Size :\t%d %d\n", width/ThreadsX, height/ThreadsY);
	printf("Block Size :\t%d %d\n", ThreadsX, ThreadsY);

	// copy data from host to device
	cudaCheckAPIError( cudaMemcpy( Voronoi_d, Voronoi_h, voronoi_size, cudaMemcpyHostToDevice) );
	free( Voronoi_h );

	executeVoronoiBM();
	saveVoronoi("voronoi.png");

/*	
	need to first copy data from device to host (output)
	printf("Displaying Output to check..\n");
        glutInit( &argc, argv );
        glutInitDisplayMode( GLUT_SINGLE | GLUT_RGBA );
        glutInitWindowSize( 512, 512 );
        glutCreateWindow( "Voronoi" );
        glutKeyboardFunc(Key);
        glutDisplayFunc(Draw);
        glutMainLoop();
*/
	cleanup();
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
