
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <vector>
#include <stdio.h>
#include <algorithm> 
#include <string>
#include <thrust/device_vector.h>

template<typename  T>
struct Vec_CUDA {
	T* v;
	int size;

	__device__
		Vec_CUDA() { size = 0; }

	__device__
		Vec_CUDA(int _size) : size(_size) {
		v = new T[size];
	}

	__device__
		Vec_CUDA(int _size, T value) : size(_size) {
		v = new T[_size];
		for (size_t i = 0; i < _size; i++)
		{
			v[i] = value;
		}
	}

	__device__
		Vec_CUDA(const Vec_CUDA& r) {
		size = r.size;
		v = new T[r.size];
		std::memcpy(v, r.v, size * sizeof(T));
	}

	__device__
		~Vec_CUDA() { delete[] v; }
	__device__
		Vec_CUDA& operator=(Vec_CUDA& r) {
			if (size > 0)
				delete[] v;
			size = r.size;
			v = new T[r.size];
			std::memcpy(v, r.v, size * sizeof(T));
			return *this;
		}
	__device__
		Vec_CUDA operator+(Vec_CUDA r) {
		Vec_CUDA rv(size);
		for (size_t i = 0; i < size; i++)
			rv.v[i] = v[i] + r.v[i];
		return rv;
	}
	__device__
		Vec_CUDA operator-(Vec_CUDA r) {
		Vec_CUDA rv(size);
		for (size_t i = 0; i < size; i++)
			rv.v[i] = v[i] - r.v[i];
		return rv;
	}
	__device__
		Vec_CUDA operator*(Vec_CUDA r) {
		Vec_CUDA rv(size);
		for (size_t i = 0; i < size; i++)
			rv.v[i] = v[i] * r.v[i];
		return rv;
	}
	template<typename  R>
	__device__
		Vec_CUDA operator*(R r) {
		Vec_CUDA rv(size);
		for (size_t i = 0; i < size; i++)
			rv.v[i] = v[i] * r;
		return rv;
	}
	__device__
		Vec_CUDA operator/(Vec_CUDA r) {
		Vec_CUDA rv(size);
		for (size_t i = 0; i < size; i++)
			rv.v[i] = v[i] / r.v[i];
		return rv;
	}
	template<typename  R>
	__device__
		Vec_CUDA operator/(R r) {
		Vec_CUDA rv(size);
		for (size_t i = 0; i < size; i++)
			rv.v[i] = v[i] / r;
		return rv;
	}

	//__device__
	//T& operator[](int i) { return v[i]; }

	__device__
		T getValue(int i) {
		return v[i];
	}
};

template<typename  T>
struct Vec {
	std::vector<T> v;
	int size;

	Vec() {}
	Vec(int _size) : v(std::vector<T>(_size)), size(_size) {}
	Vec(std::vector<T> _v) : v(_v), size(_v.size()) {}
	Vec(int _size, T value) : v(std::vector<T>(_size, value)), size(_size) {}
	Vec operator+(Vec r) {
		Vec rv(v.size());
		for (size_t i = 0; i < v.size(); i++)
			rv.v[i] = v[i] + r.v[i];
		return rv;
	}
	Vec operator-(Vec r) {
		Vec rv(v.size());
		for (size_t i = 0; i < v.size(); i++)
			rv.v[i] = v[i] - r.v[i];
		return rv;
	}
	Vec operator*(Vec r) {
		Vec rv(v.size());
		for (size_t i = 0; i < v.size(); i++)
			rv.v[i] = v[i] * r.v[i];
		return rv;
	}
	template<typename  R>
	Vec operator*(R r) {
		Vec rv(v.size());
		for (size_t i = 0; i < v.size(); i++)
			rv.v[i] = v[i] * r;
		return rv;
	}
	Vec operator/(Vec r) {
		Vec rv(v.size());
		for (size_t i = 0; i < v.size(); i++)
			rv.v[i] = v[i] / r.v[i];
		return rv;
	}
	template<typename  R>
	Vec operator/(R r) {
		Vec rv(v.size());
		for (size_t i = 0; i < v.size(); i++)
			rv.v[i] = v[i] / r;
		return rv;
	}
	T& operator[](int i) { return v[i]; }

	static Vec<T> cross(Vec<T> v1, Vec<T> v2) {
		T vx = v1[1] * v2[2] - v1[2] * v2[1];
		T vy = v1[2] * v2[0] - v1[0] * v2[2];
		T vz = v1[0] * v2[1] - v1[1] * v2[0];
		return Vec<T>({ vx, vy, vz });
	}

	static Vec<T> normalize(Vec<T> v) {
		T squareSum = 0;
		for (size_t i = 0; i < v.size; i++)
			squareSum += pow(v[i], 2);
		return v / sqrt(squareSum);
	}

	static Vec<T> min(Vec<T> v1, Vec<T> v2) {
		Vec<T> v(v1.size);
		for (size_t i = 0; i < v.size; i++)
			v[i] = v1[i] < v2[i] ? v1[i] : v2[i];
		return v;
	}

	static Vec<T> max(Vec<T> v1, Vec<T> v2) {
		Vec<T> v(v1.size);
		for (size_t i = 0; i < v.size; i++)
			v[i] = v1[i] > v2[i] ? v1[i] : v2[i];
		return v;
	}

	std::string to_string() {
		std::string s("");
		for (size_t i = 0; i < size; i++)
			s += std::to_string(v[i]) + " ";
		return s;
	}

	T sum() {
		T sum = 0;
		for (size_t i = 0; i < size; i++)
			sum = sum + v[i];
		return sum;
	}

	void push_back(T element) {
		v.push_back(element);
		size++;
	}
};

template<typename  T>
struct Vec3 {
	T x, y, z;

	Vec3() {}

	Vec3(T _x, T _y, T _z) : x(_x), y(_y), z(_z) {}

	Vec3 operator+(Vec3 r) { return Vec3(x + r.x, y + r.y, z + r.z); }
	Vec3 operator-(Vec3 r) { return Vec3(x - r.x, y - r.y, z - r.z); }
	Vec3 operator*(Vec3 r) { return Vec3(x * r.x, y * r.y, z * r.z); }

	template<typename  R>
	Vec3 operator*(R r) { return Vec3(x * r, y * r, z * r); }

	Vec3 operator/(Vec3 r) { return Vec3(x / r.x, y / r.y, z / r.z); }

	template<typename  R>
	Vec3 operator/(R r) { return Vec3(x / r, y / r, z / r); }

	//__device__
	//T& operator[](int i) { return v[i]; }

	static Vec3<T> cross(Vec3<T> v1, Vec3<T> v2) {
		T x = v1.y * v2.z - v1.z * v2.y;
		T y = v1.z * v2.x - v1.x * v2.z;
		T z = v1.x * v2.y - v1.y * v2.x;
		return Vec3(x, y, z);
	}

	static Vec3<T> normalize(Vec3<T> v) {
		T squareSum = 0;
		squareSum += pow(v.x, 2);
		squareSum += pow(v.y, 2);
		squareSum += pow(v.z, 2);
		return v / sqrt(squareSum);
	}

	static Vec3<T> min(Vec3<T> v1, Vec3<T> v2) {
		T x = v1.x < v2.x ? v1.x : v2.x;
		T y = v1.y < v2.y ? v1.y : v2.y;
		T z = v1.z < v2.z ? v1.z : v2.z;
		return Vec3_CUDA(x, y, z);
	}

	static Vec3<T> max(Vec3<T> v1, Vec3<T> v2) {
		T x = v1.x > v2.x ? v1.x : v2.x;
		T y = v1.y > v2.y ? v1.y : v2.y;
		T z = v1.z > v2.z ? v1.z : v2.z;
		return Vec3_CUDA(x, y, z);
	}

	std::string to_string() {
		std::string s("");
		s += std::to_string(x) + " " + std::to_string(y) + " " + std::to_string(z);
		return s;
	}
};

template<typename  T>
struct Vec3_CUDA {
	T x, y, z;

	__device__
		Vec3_CUDA() {}

	__device__
		Vec3_CUDA(T _x, T _y, T _z) : x(_x), y(_y), z(_z) {}

	__device__
		Vec3_CUDA(Vec3<T> v) : x(v.x), y(v.y), z(v.z) {}

	__device__
		Vec3_CUDA operator+(Vec3_CUDA r) { return Vec3_CUDA(x + r.x, y + r.y, z + r.z); }
	__device__
		Vec3_CUDA operator-(Vec3_CUDA r) { return Vec3_CUDA(x - r.x, y - r.y, z - r.z); }
	__device__
		Vec3_CUDA operator*(Vec3_CUDA r) { return Vec3_CUDA(x * r.x, y * r.y, z * r.z); }

	template<typename  R>
	__device__
		Vec3_CUDA operator*(R r) { return Vec3_CUDA(x * r, y * r, z * r); }

	__device__
		Vec3_CUDA operator/(Vec3_CUDA r) { return Vec3_CUDA(x / r.x, y / r.y, z / r.z); }

	template<typename  R>
	__device__
		Vec3_CUDA operator/(R r) { return Vec3_CUDA(x / r, y / r, z / r); }

	//__device__
	//T& operator[](int i) { return v[i]; }

	__device__
		static Vec3_CUDA<T> cross(Vec3_CUDA<T> v1, Vec3_CUDA<T> v2) {
		T x = v1.y * v2.z - v1.z * v2.y;
		T y = v1.z * v2.x - v1.x * v2.z;
		T z = v1.x * v2.y - v1.y * v2.x;
		return Vec3_CUDA(x, y, z);
	}

	__device__
		static Vec3_CUDA<T> normalize(Vec3_CUDA<T> v) {
		T squareSum = 0;
		squareSum += pow(v.x, 2);
		squareSum += pow(v.y, 2);
		squareSum += pow(v.z, 2);
		return v / sqrt(squareSum);
	}

	__device__
		static Vec3_CUDA<T> min(Vec3_CUDA<T> v1, Vec3_CUDA<T> v2) {
		T x = v1.x < v2.x ? v1.x : v2.x;
		T y = v1.y < v2.y ? v1.y : v2.y;
		T z = v1.z < v2.z ? v1.z : v2.z;
		return Vec3_CUDA(x, y, z);
	}

	__device__
		static Vec3_CUDA<T> max(Vec3_CUDA<T> v1, Vec3_CUDA<T> v2) {
		T x = v1.x > v2.x ? v1.x : v2.x;
		T y = v1.y > v2.y ? v1.y : v2.y;
		T z = v1.z > v2.z ? v1.z : v2.z;
		return Vec3_CUDA(x, y, z);
	}

	std::string to_string() {
		std::string s("");
		s += std::to_string(x) + " " + std::to_string(y) + " " + std::to_string(z);
		return s;
	}
};

__device__
float dMax(float a, float b) {
	return a > b ? a : b;
}

__device__
float dMin(float a, float b) {
	return a < b ? a : b;
}

__device__
float square(float v) {
	return v * v;
}

__managed__ int voxResolutionGlob;

class Volume {
	//-------------------------------------------------------------
	unsigned int voxResolution;
	unsigned int maxVoxValue = 0;
	int* voxels;
public:
	Volume(char* filename) {
		voxResolution = 0;
		std::vector<unsigned char> fileVoxels;
		FILE* file = fopen(filename, "rb");
		if (!file) {
			std::cerr << "File " << filename << " not found\n";
			return;
		}
		if (fscanf(file, "%d", &voxResolution) != 1) return;
		int nVoxels = voxResolution * voxResolution * voxResolution;
		voxels = new int[nVoxels];
		fileVoxels.resize(nVoxels);
		fread(&fileVoxels[0], 1, nVoxels, file);
		for (size_t i = 0; i < nVoxels; i++)
		{
			voxels[i] = (int)fileVoxels[i];
			if (voxels[i] > maxVoxValue) maxVoxValue = voxels[i];
		}
	}

	~Volume() { delete[] voxels; }

	int* getVoxels() { return voxels; }

	__device__
		int voxelValueAt(int x, int y, int z, int* voxels2) {
		if (x == voxResolutionGlob) x--;
		if (y == voxResolutionGlob) y--;
		if (z == voxResolutionGlob) z--;
		return voxels2[(int)square(voxResolutionGlob) * z + voxResolutionGlob * y + x];
	}

	__device__
		float volumeValueAt(Vec3_CUDA<float> position, int* voxels2) { //Trilinear interpolation
		position = position * (voxResolutionGlob - 1);
		int xf = std::floor(position.x);
		int xc = std::floor(position.x + 1.0f);
		int yf = std::floor(position.y);
		int yc = std::floor(position.y + 1.0f);
		int zf = std::floor(position.z);
		int zc = std::floor(position.z + 1.0f);

		float dx = (position.x - xf) / (xc - xf);
		float dy = (position.y - yf) / (yc - yf);
		float dz = (position.z - zf) / (zc - zf);

		float c00 = voxelValueAt(xf, yf, zf, voxels2) * (1 - dx) + voxelValueAt(xc, yf, zf, voxels2) * dx;
		float c01 = voxelValueAt(xf, yf, zc, voxels2) * (1 - dx) + voxelValueAt(xc, yf, zc, voxels2) * dx;
		float c10 = voxelValueAt(xf, yc, zf, voxels2) * (1 - dx) + voxelValueAt(xc, yc, zf, voxels2) * dx;
		float c11 = voxelValueAt(xf, yc, zc, voxels2) * (1 - dx) + voxelValueAt(xc, yc, zc, voxels2) * dx;

		float c0 = c00 * (1 - dy) + c10 * dy;
		float c1 = c01 * (1 - dy) + c11 * dy;

		return (c0 * (1 - dz) + c1 * dz) / 255;
	}

	int getVoxResolution() { return voxResolution; }
	__device__
	int getMaxVoxValue() { return maxVoxValue; }
};

struct Dnum_CUDA {
	float f;
	Vec_CUDA<float> d;
	static const int size = 24;

	__device__ Dnum_CUDA() {}
	__device__ Dnum_CUDA(float f0) : f(f0), d(Vec_CUDA<float>(size, 0.0f)) {}
	__device__ Dnum_CUDA(float f0, Vec_CUDA<float> d0) : f(f0), d(d0) {}
	__device__ Dnum_CUDA(float f, int index) : f(f) {
		d = Vec_CUDA<float>(size, 0.0f);
		d.v[index] = 1.0f;
	}
	__device__ Dnum_CUDA(const Dnum_CUDA& r) : f(r.f), d(r.d) {}
	__device__
		Dnum_CUDA& operator=(Dnum_CUDA& r) {
		f = r.f;
		d = r.d;
		return *this;
	}
	__device__ Dnum_CUDA operator+(Dnum_CUDA r) { return Dnum_CUDA(f + r.f, d + r.d); }
	__device__ Dnum_CUDA operator-(Dnum_CUDA r) { return Dnum_CUDA(f - r.f, d - r.d); }
	__device__ Dnum_CUDA operator*(Dnum_CUDA r) { return Dnum_CUDA(f * r.f, r.d * f + d * r.f); }
	__device__ Dnum_CUDA operator/(Dnum_CUDA r) { return Dnum_CUDA(f / r.f, (d * r.f - r.d * f) / (r.f * r.f)); }
	__device__ static Dnum_CUDA Exp(Dnum_CUDA g) { return Dnum_CUDA(expf(g.f), g.d * expf(g.f)); }
	std::string to_string() {
		std::string s("");
		s += "value: " + std::to_string(f);
		for (size_t i = 0; i < size; i++)
			s += ", d_p" + std::to_string(i) + ": " + std::to_string(d.v[i]);
		return s;
	}
};

struct Dnum {
	float f;
	Vec<float> d;
	static const int size = 24;

	Dnum() : f(0.0f), d(Vec<float>(size, 0.0f)) {}
	Dnum(float f0) : f(f0), d(Vec<float>(size, 0.0f)) {}
	Dnum(float f0, Vec<float> d0) : f(f0), d(d0) {}
	Dnum(float f, int index) : f(f) {
		d = Vec<float>(size, 0.0f);
		d.v[index] = 1.0f;
	}
	Dnum operator+(Dnum r) { return Dnum(f + r.f, d + r.d); }
	Dnum operator-(Dnum r) { return Dnum(f - r.f, d - r.d); }
	Dnum operator*(Dnum r) { return Dnum(f * r.f, r.d * f + d * r.f); }
	Dnum operator/(Dnum r) { return Dnum(f / r.f, (d * r.f - r.d * f) / (r.f * r.f)); }
	static Dnum Exp(Dnum g) { return Dnum(expf(g.f), g.d * expf(g.f)); }
	std::string to_string() {
		std::string s("");
		s += "value: " + std::to_string(f);
		for (size_t i = 0; i < size; i++)
			s += ", d_p" + std::to_string(i) + ": " + std::to_string(d.v[i]);
		return s;
	}
};

__managed__ float* params; // ext( 4 r, 4 g, 4 b ), emit( 4 r, 4 g, 4 b )

void initializeParams() {
	cudaMallocManaged(&params, Dnum::size * sizeof(float));
	float paramValues[] = { 0.0f, 1.0f, 1.0f, 0.0f,
							0.0f, 1.0f, 1.0f, 0.0f,
							0.0f, 1.0f, 1.0f, 0.0f,
							-2.0f, 2.0f, 1.0f, 0.0f,
							-10.0f, 10.0f, 1.0f, 0.0f,
							0.0f, 1.0f, 1.0f, 0.0f };
	for (size_t i = 0; i < Dnum::size; i++)
	{
		params[i] = paramValues[i];
	}
}

__device__
Dnum_CUDA sigmoid(Dnum_CUDA value, Dnum_CUDA p0, Dnum_CUDA p1, Dnum_CUDA p2, Dnum_CUDA p3) {
	return p0 + (p1 / (Dnum_CUDA(1) + Dnum_CUDA::Exp(p2 * -1 * (value - p3))));
}

__device__
Dnum_CUDA computeRadiance(bool& isBackground, Volume* volume, int* voxels, Vec3_CUDA<float> eye, Vec3_CUDA<float> dir, float entry, float exit, float dt, 
	int waveLength) {

	Dnum_CUDA L(0.0f);
	Vec3_CUDA<float> r = eye;
	Dnum_CUDA opt(0.0f);
	int index = 0;
	for (float t = entry; t < exit; t += dt) {
		r = eye + dir * t;
		if (r.x > 1 || r.x < 0 || r.y > 1 || r.y < 0 || r.z > 1 || r.z < 0) continue;
		if (volume->volumeValueAt(r, voxels) > (float)80/255) {
			isBackground = false;
			opt = opt + sigmoid(volume->volumeValueAt(r, voxels), Dnum_CUDA(params[waveLength * 4], waveLength * 4), 
				Dnum_CUDA(params[waveLength * 4 + 1], waveLength * 4 + 1), Dnum_CUDA(params[waveLength * 4 + 2], waveLength * 4 + 2),
					Dnum_CUDA(params[waveLength * 4 + 3], waveLength * 4 + 3)) * dt;
			L = L + sigmoid(volume->volumeValueAt(r, voxels), Dnum_CUDA(params[12 + waveLength * 4], 12 + waveLength * 4),
				Dnum_CUDA(params[12 + waveLength * 4 + 1], 12 + waveLength * 4 + 1), Dnum_CUDA(params[12 + waveLength * 4 + 2], 12 + waveLength * 4 + 2),
				Dnum_CUDA(params[12 + waveLength * 4 + 3], 12 + waveLength * 4 + 3)) * dt *Dnum_CUDA::Exp(opt * -1);
		}
		index++;
	}
	return L;
}

__global__
void populateC(float* radianceValues, int windowResolution, Vec3<float> _lookat, Vec3<float> _right, Vec3<float> _up, Vec3<float> _eye,
	Volume* volume, int* voxels, float dt, int wavelengthSize)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < windowResolution * windowResolution * wavelengthSize) {
		Vec3_CUDA<float> lookat(_lookat);
		Vec3_CUDA<float> right(_right);
		Vec3_CUDA<float> up(_up);
		Vec3_CUDA<float> eye(_eye);
		int i = index / square(windowResolution);
		int x = index % (int)square(windowResolution) / windowResolution;
		int y = index % (int)square(windowResolution) % windowResolution;
		Vec3_CUDA<float> p = lookat + right * (-1 + x * (float)2 / (windowResolution - 1)) + up * (-1 + y * (float)2 / (windowResolution - 1));
		Vec3_CUDA<float> dir = Vec3_CUDA<float>::normalize(p - eye);
		Vec3_CUDA<float> t0 = (Vec3_CUDA<float>(0.0f, 0.0f, 0.0f) - eye) / dir, t1 = (Vec3_CUDA<float>(1.0f, 1.0f, 1.0f) - eye) / dir;
		Vec3_CUDA<float> ti = Vec3_CUDA<float>::min(t0, t1), to = Vec3_CUDA<float>::max(t0, t1);
		float entry = dMax(dMax(ti.x, ti.y), ti.z), exit = dMin(dMin(to.x, to.y), to.z);
		bool isBackground = true;
		Dnum_CUDA d = computeRadiance(isBackground, volume, voxels, eye, dir, entry, exit, dt, i);
		if(isBackground) radianceValues[index * (Dnum::size + 1)] = 10000.0;
		else {
			radianceValues[index * (Dnum::size + 1)] = d.f;
			for (size_t i = 0; i < Dnum_CUDA::size; i++)
			{
				radianceValues[index * (Dnum::size + 1) + i + 1] = d.d.v[i];
			}
		}		
	}	
}

Dnum calculateC(int windowResolution, Vec3<float> lookat, Vec3<float> right, Vec3<float> up, Vec3<float> eye, Volume* volume, float dt)
{
	const int wavelengthSize = 3;
	const int radianceSize = pow(windowResolution, 2) * wavelengthSize;
	int* voxels;
	float* radianceValues;

	cudaMallocManaged(&voxels, pow(voxResolutionGlob, wavelengthSize) * sizeof(int));
	cudaMallocManaged(&radianceValues, radianceSize * (Dnum::size + 1) * sizeof(float));

	for (size_t i = 0; i < pow(voxResolutionGlob, wavelengthSize); i++)
	{
		voxels[i] = volume->getVoxels()[i];
	}

	int blockSize = 256;
	int numBlocks = (radianceSize + blockSize - 1) / blockSize;
	populateC << <numBlocks, blockSize >> > (radianceValues, windowResolution, lookat, right, up, eye, volume, voxels, dt, wavelengthSize);

	cudaDeviceSynchronize();

	Vec<Dnum> radiance(0);
	int index = 0;
	for (size_t i = 0; i < radianceSize; i++) {
		if (radianceValues[i * (Dnum::size + 1)] != 10000.0) {
			radiance.push_back(Dnum(radianceValues[i * (Dnum::size + 1)]));
			for (size_t j = 0; j < Dnum::size; j++)
			{
				radiance.v[index].d.v[j] = radianceValues[i * (Dnum::size + 1) + (j + 1)];
			}
			index++;
		}		
	}
	float max[wavelengthSize];
	float min[wavelengthSize];
	for (size_t j = 0; j < wavelengthSize; j++)
	{
		max[j] = radiance[j * radiance.size / wavelengthSize].f;
		min[j] = radiance[j * radiance.size / wavelengthSize].f;
		for (size_t i = 0; i < radiance.size / wavelengthSize; i++)
		{
			if (radiance[j * radiance.size / wavelengthSize + i].f > max[j]) max[j] = radiance[j * radiance.size / wavelengthSize + i].f;
			if (radiance[j * radiance.size / wavelengthSize + i].f < min[j]) min[j] = radiance[j * radiance.size / wavelengthSize + i].f;
		}
	}
	std::cout << "maxL = {" << max[0];
	for (size_t j = 1; j < 3; j++)
	{
		std::cout << "," << (wavelengthSize > j ? max[j] : max[0]);
	}
	std::cout << "};" << std::endl;

	std::cout << "minL = {" << min[0];
	for (size_t j = 1; j < 3; j++)
	{
		std::cout << "," << (wavelengthSize > j ? min[j] : min[0]);
	}
	std::cout << "};" << std::endl;
	std::cout << "params = {" << params[0];
	for (size_t i = 1; i < Dnum::size; i++)
	{
		std::cout << ',' << params[i];
	}
	std::cout << "};" << std::endl << std::endl;
	for (size_t j = 0; j < wavelengthSize; j++)
	{
		Vec<Dnum> radianceTemp(0);
		for (size_t i = 0; i < radiance.size / wavelengthSize; i++)
		{
			radianceTemp.push_back(radiance[j * (radiance.size / wavelengthSize) + i]);
		}
		Dnum avgTemp = radianceTemp.sum() / radianceTemp.size;
		radianceTemp = radianceTemp / avgTemp;
		avgTemp = avgTemp / avgTemp;
		Dnum maxAbs = radianceTemp[0];
		avgTemp = avgTemp / (avgTemp * Dnum(2));
		for (size_t i = 0; i < radianceTemp.size; i++)
		{
			radianceTemp[i] = radianceTemp[i] - Dnum(1);
			if (abs(radianceTemp[i].f) > abs(maxAbs.f)) maxAbs = radianceTemp[i];
		}
		radianceTemp = radianceTemp / (maxAbs * 2);
		for (size_t i = 0; i < radianceTemp.size; i++)
		{
			radianceTemp[i] = radianceTemp[i] + Dnum(0.5);
			radiance[j * (radiance.size / wavelengthSize) + i] = radianceTemp[i];
		}
	}
	Dnum C(0);
	Dnum avgTemp = radiance.sum() / radiance.size;
	for (size_t i = 0; i < radiance.size; i++)
	{
		C = C + (radiance[i] - avgTemp) * (radiance[i] - avgTemp);
	}
	C = C / radiance.size;
	return C;
}

void modifyParams(Dnum C, float lrate)
{
	for (size_t i = 0; i < Dnum::size; i++)
		params[i] += C.d.v[i] * lrate;
}

#include <chrono>

int main()
{
	initializeParams();
	float lrate = 10.0f;
	auto volume = new Volume("C:\\Users\\ungbo\\Desktop\\BME\\Önlab\\2. Félév\\head128.vox");
	voxResolutionGlob = volume->getVoxResolution();
	const float eyeRadius = 4.0f;
	float dt = 0.5f / volume->getVoxResolution();
	const int windowResolution = 50;
	float eyeAngle = 90.0f;
	Vec3<float> lookat(0.5f, 0.5f, 0.5f);
	Vec3<float> up(0.0f, 1.0f, 0.0f);
	Vec3<float> dEye(cosf(eyeAngle), 0.0f, sinf(eyeAngle));
	Vec3<float> right = Vec3<float>::cross(up, dEye);
	Vec3<float> eye = lookat + dEye * eyeRadius;
	int period = 30;
	for (size_t i = 0; i < 5 * period; i++)
	{
		auto start = std::chrono::high_resolution_clock::now();
		if (i % period == period - 1) lrate *= 0.1;
		Dnum C = calculateC(windowResolution, lookat, right, up, eye, volume, dt);
		modifyParams(C, lrate);
		std::cout << C.to_string() << std::endl << std::endl;
		std::cout << "learning rate: " << lrate << std::endl << std::endl;
		auto stop = std::chrono::high_resolution_clock::now();
		std::cout << "processing time: " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1e6 << " s" << std::endl;
	}
	cudaFree(params);
	delete volume;
	return 0;
}