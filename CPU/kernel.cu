
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <vector>
#include <stdio.h>
#include <algorithm> 
#include <string>

template<typename  T>
struct Vec {
	std::vector<T> v;
	int size;

	Vec() {}
	Vec(int _size) : v(std::vector<T>(_size)), size(_size) {}
	Vec(std::vector<T> _v) : v(_v), size(_v.size()) {}
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
		size++;
		v.push_back(element);
	}
};

struct Dnum {
	float f;
	Vec<float> d;
	static const int size = 24;

	Dnum() {}
	Dnum(float f0, Vec<float> d0 = Vec<float>(size)) : f(f0), d(d0) {}
	Dnum operator+(Dnum r) { return Dnum(f + r.f, d + r.d); }
	Dnum operator-(Dnum r) { return Dnum(f - r.f, d - r.d); }
	Dnum operator*(Dnum r) { return Dnum(f * r.f, r.d * f + d * r.f); }
	Dnum operator/(Dnum r) { return Dnum(f / r.f, (d * r.f - r.d * f) / (r.f * r.f)); }
	static Dnum Exp(Dnum g) { return Dnum(expf(g.f), g.d * expf(g.f)); }
	std::string to_string() {
		std::string s("");
		s += "value: " + std::to_string(f);
		for (size_t i = 0; i < size; i++)
			s += ", d_p" + std::to_string(i) + ": " + std::to_string(d[i]);
		return s;
	}
};

class Volume {
	//-------------------------------------------------------------
	unsigned int voxResolution;
	unsigned int maxVoxValue = 0;
	std::vector<int> voxels;
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
		voxels.resize(nVoxels);
		fileVoxels.resize(nVoxels);
		fread(&fileVoxels[0], 1, nVoxels, file);
		for (size_t i = 0; i < nVoxels; i++)
		{
			voxels[i] = (int)fileVoxels[i];
			if (voxels[i] > maxVoxValue) maxVoxValue = voxels[i];
		}
	}

	int voxelValueAt(int x, int y, int z) {
		if (x == voxResolution) x--;
		if (y == voxResolution) y--;
		if (z == voxResolution) z--;
		return voxels[pow(voxResolution, 2) * z + voxResolution * y + x]; //TODO jók-e a dimenziók
	}

	float volumeValueAt(Vec<float> position) { //Trilinear interpolation
		for (size_t i = 0; i < 3; i++)
			if (position[i] < 0.0f || position[i] > 1.0f) throw "Invalid voxel coordinates";
		position = position * (voxResolution - 1);

		int xf = std::floor(position[0]);
		int xc = std::floor(position[0] + 1.0f);
		int yf = std::floor(position[1]);
		int yc = std::floor(position[1] + 1.0f);
		int zf = std::floor(position[2]);
		int zc = std::floor(position[2] + 1.0f);

		float dx = (position[0] - xf) / (xc - xf);
		float dy = (position[1] - yf) / (yc - yf);
		float dz = (position[2] - zf) / (zc - zf);

		float c00 = voxelValueAt(xf, yf, zf) * (1 - dx) + voxelValueAt(xc, yf, zf) * dx;
		float c01 = voxelValueAt(xf, yf, zc) * (1 - dx) + voxelValueAt(xc, yf, zc) * dx;
		float c10 = voxelValueAt(xf, yc, zf) * (1 - dx) + voxelValueAt(xc, yc, zf) * dx;
		float c11 = voxelValueAt(xf, yc, zc) * (1 - dx) + voxelValueAt(xc, yc, zc) * dx;

		float c0 = c00 * (1 - dy) + c10 * dy;
		float c1 = c01 * (1 - dy) + c11 * dy;

		return (c0 * (1 - dz) + c1 * dz)/maxVoxValue;
	}

	int getVoxResolution() { return voxResolution; }
	int getMaxVoxValue() { return maxVoxValue; }
	std::vector<int> getVoxels() { return voxels; }
};

Dnum createParam(float value, int paramIndex) {
	std::vector<float> v(Dnum::size, 0.0f);
	v[paramIndex] = 1.0f;
	return Dnum(value, v);
}

Vec<Dnum> params(Dnum::size); // ext( 4 r, 4 g, 4 b ), emit( 4 r, 4 g, 4 b ) 

void initializeParams() {
	float paramValues[] = {0.0f, 1.0f, 1.0f, 0.0f,
							0.0f, 1.0f, 1.0f, 0.0f, 
							0.0f, 1.0f, 1.0f, 0.0f, 
							0.0f, 1.0f, 1.0f, 0.0f,
							0.0f, 1.0f, 1.0f, 0.0f,
							0.0f, 1.0f, 1.0f, 0.0f };
	for (size_t i = 0; i < Dnum::size; i++)
	{
		params[i] = createParam(paramValues[i], i);
	}
}

Dnum sigmoid(Dnum value, Dnum p0, Dnum p1, Dnum p2, Dnum p3) {
	return p0 + (p1 / (Dnum(1) + Dnum::Exp(p2 * Dnum(-1) * (value - p3))));
}

Dnum computeRadiance(bool& isBackground ,Volume* volume, Vec<float> eye, Vec<float> dir, float entry, float exit, float dt, int waveLength) {

	Dnum L(0.0f);
	Vec<float> r = eye;
	Dnum opt(0.0f);
	for (float t = entry; t < exit; t += dt) {
		r = eye + dir * t;
		if (r[0] > 1 || r[0] < 0 || r[1] > 1 || r[1] < 0 || r[2] > 1 || r[2] < 0) continue;
		if (volume->volumeValueAt(r) > (float)80/volume->getMaxVoxValue()) {
			isBackground = false;
			opt = opt + sigmoid(volume->volumeValueAt(r), params[waveLength * 4], params[waveLength * 4 + 1], params[waveLength * 4 + 2],
				params[waveLength * 4 + 3]) * dt;
			L = L + sigmoid(volume->volumeValueAt(r), params[12 + waveLength * 4], params[12 + waveLength * 4 + 1], params[12 + waveLength * 4 + 2],
				params[12 + waveLength * 4 + 3]) * dt *Dnum::Exp((opt) * -1);
		}
	}
	return L;
}

Dnum plainSigmoid(Dnum value) 
{
	return Dnum(1) / (Dnum(1) + Dnum::Exp(Dnum(-1) * value));
}

Dnum calculateC(int windowResolution, Vec<float> lookat, Vec<float> right, Vec<float> up, Vec<float> eye, Volume* volume, float dt)
{
	const int wavelengthSize = 3;
	Vec<Dnum> radiance(0);
	for (size_t i = 0; i < wavelengthSize; i++)
	{
		for (int x = 0; x < windowResolution; x++)
		{
			for (int y = 0; y < windowResolution; y++)
			{
				Vec<float> p = lookat + right * (-1 + x * (float)2 / (windowResolution - 1)) + up * (-1 + y * (float)2 / (windowResolution - 1));
				Vec<float> dir = Vec<float>::normalize(p - eye);
				Vec<float> t0 = (Vec<float>({ 0.0f, 0.0f, 0.0f }) - eye) / dir, t1 = (Vec<float>({ 1.0f, 1.0f, 1.0f }) - eye) / dir;
				Vec<float> ti = Vec<float>::min(t0, t1), to = Vec<float>::max(t0, t1);
				float entry = std::max(std::max(ti[0], ti[1]), ti[2]), exit = std::min(std::min(to[0], to[1]), to[2]);
				bool isBackground = true;
				Dnum d(computeRadiance(isBackground, volume, eye, dir, entry, exit, dt, i));
				if (!isBackground) radiance.push_back(d);
			}
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
	std::cout << "};"<<std::endl;

	std::cout << "minL = {" << min[0];
	for (size_t j = 1; j < 3; j++)
	{
		std::cout << "," << (wavelengthSize > j ? min[j] : min[0]);
	}
	std::cout << "};" << std::endl;
	std::cout << "params = {" << params[0].f;
	for (size_t i = 1; i < params.size; i++)
	{
		std::cout << ',' << params[i].f;
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
			//radianceTemp[i] = radianceTemp[i] / (1.0f / 255.0f) * (1.0f / 255.0f);
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
	{
		params[i].f += C.d[i] * lrate;
	}
}

#include "gnuplot-iostream.h";
#include <random>;
#include <numeric>;
#include <chrono>

int main()
{
	initializeParams();
	float lrate = 10.0f;
	auto volume = new Volume("C:\\Users\\ungbo\\Desktop\\BME\\Önlab\\2. Félév\\head128.vox");
	const float eyeRadius = 4.0f;
	float dt = 0.5f / volume->getVoxResolution();
	const int windowResolution = 20;
	float eyeAngle = 90.0f;
	Vec<float> lookat({ 0.5f, 0.5f, 0.5f });
	Vec<float> up({ 0.0f, 1.0f, 0.0f });
	Vec<float> dEye({ cosf(eyeAngle), 0.0f, sinf(eyeAngle) });
	Vec<float> right = Vec<float>::cross(up, dEye);
	Vec<float> eye = lookat + dEye * eyeRadius;
	Dnum C(0);
	int period = 30;
	for (size_t i = 0; i < period * 5; i++)
	{
		auto start = std::chrono::high_resolution_clock::now();
		if (i % period == period - 1) lrate *= 0.1;
		C = calculateC(windowResolution, lookat, right, up, eye, volume, dt);
		modifyParams(C, lrate);
		std::cout << C.to_string() << std::endl << std::endl;
		std::cout << "learning rate: " << lrate << std::endl << std::endl;
		auto stop = std::chrono::high_resolution_clock::now();
		std::cout << "processing time: " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1e6 <<" s"<<std::endl ;
	}
	//std::cout << calculateC(100, lookat, right, up, eye, volume, dt).f;
	return 0;
}
//10 - 0,2
//20 - 1
//30 - 2,3
//50 - 6,3
//100 - 

//600
//10 - 0.051455
//20 - 0.062739
//30 - 0.070298
//50 - 0.071285
//100 - 0.074146
