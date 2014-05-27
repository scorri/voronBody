//http://www.bu.edu/pasi/files/2011/07/Lab5.pdf
//http://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
/*
	Sort Voronoi using Morton Code
*/
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cmath>
#include <thrust/sort.h>

const int n = 4;
const int uNumVoronoiPts = 16;

__host__ __device__ struct Color{
	int blue, green, red;
	int dist;
	int index;
	int morton;
	void print()
	{
		printf("%d\t%d\t%d\t%d\t%d\t%d\n", index, morton, blue, green, red, dist);
	}
};

__host__ __device__ bool operator<(const Color &lhs, const Color &rhs) 
{
	return lhs.morton < rhs.morton;
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

int main(void)
{
	thrust::device_vector<Color> cd;
	thrust::host_vector<Color> ch;
	
	printf("Unsorted\n");
	printf("index\tmorton\tblue\tgreen\tred\tdist\n");
	int idx = 0;
	for (int y = 0; y < n; y++)	//row major
	{
		for(int x = 0; x < n; x++)
		{
			Color c;
			//std::cout << "x " << x << " y " << y << std::endl;
			//std::cout << "ID: " << idx <<" Morton " << EncodeMorton2(x, y) << std::endl;

			c.blue = 25 + 204 * (rand()%256)/255;
			c.green = 25 + 204 * (rand()%256)/255;
			c.red = 25 + 204 * (rand()%256)/255;
			c.dist = rand();
			c.index = idx;
			c.morton = EncodeMorton2(x, y);
			c.print();
			ch.push_back(c);
			idx++;
		}
		//std::cout << std::endl;
	}
	
	cd = ch;
	thrust::sort(cd.begin(), cd.end());
	ch = cd;

	printf("\nSorted Morton Codes(Second Columm)\n");
	printf("index\tmorton\tblue\tgreen\tred\tdist\n");
	for(unsigned int i = 0; i < ch.size(); i++)
		ch[i].print();

	system("pause");
	return 0;
}
