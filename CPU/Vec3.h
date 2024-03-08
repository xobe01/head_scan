#pragma once
template<typename  T>
struct Vec3 {
	int x, y, z;

	__device__
		Vec() {}

	__device__
		Vec(int _size) : v(thrust::device_vector<T>(_size)), size(_size) {
		cudaMallocManaged(&v, size * sizeof(T));
	}

	__device__
		Vec(T x, T y, T z) : size(3) {
		cudaMallocManaged(&v, 3 * sizeof(T));
		v[0] = x;
		v[1] = y;
		v[2] = z;
	}

	__device__
		Vec operator+(Vec r) {
		Vec rv(v.size());
		for (size_t i = 0; i < v.size(); i++)
			v[i] = v[i] + r.v[i];
		return rv;
	}
	__device__
		Vec operator-(Vec r) {
		Vec rv(v.size());
		for (size_t i = 0; i < v.size(); i++)
			rv.v[i] = v[i] - r.v[i];
		return rv;
	}
	__device__
		Vec operator*(Vec r) {
		Vec rv(v.size());
		for (size_t i = 0; i < v.size(); i++)
			rv.v[i] = v[i] * r.v[i];
		return rv;
	}
	template<typename  R>
	__device__
		Vec operator*(R r) {
		Vec rv(v.size());
		for (size_t i = 0; i < v.size(); i++)
			rv.v[i] = v[i] * r;
		return rv;
	}
	__device__
		Vec operator/(Vec r) {
		Vec rv(v.size());
		for (size_t i = 0; i < v.size(); i++)
			rv.v[i] = v[i] / r.v[i];
		return rv;
	}
	template<typename  R>
	__device__
		Vec operator/(R r) {
		Vec rv(v.size());
		for (size_t i = 0; i < v.size(); i++)
			rv.v[i] = v[i] / r;
		return rv;
	}

	//__device__
	//T& operator[](int i) { return v[i]; }

	__device__
		static Vec<T> cross(Vec<T> v1, Vec<T> v2) {
		T vx = v1.v[1] * v2.v[2] - v1.v[2] * v2.v[1];
		T vy = v1.v[2] * v2.v[0] - v1.v[0] * v2.v[2];
		T vz = v1.v[0] * v2.v[1] - v1.v[1] * v2.v[0];
		thrust::device_vector<T> vec(3);
		vec[0] = vx;
		vec[1] = vy;
		vec[2] = vz;
		return Vec<T>(vec);
	}

	__device__
		static Vec<T> normalize(Vec<T> v) {
		T squareSum = 0;
		for (size_t i = 0; i < v.size; i++)
			squareSum += pow(v.v[i], 2);
		return v / sqrt(squareSum);
	}

	__device__
		static Vec<T> min(Vec<T> v1, Vec<T> v2) {
		Vec<T> v(v1.size);
		for (size_t i = 0; i < v.size; i++)
			v.v[i] = v1.v[i] < v2.v[i] ? v1.v[i] : v2.v[i];
		return v;
	}

	__device__
		static Vec<T> max(Vec<T> v1, Vec<T> v2) {
		Vec<T> v(v1.size);
		for (size_t i = 0; i < v.size; i++)
			v.v[i] = v1.v[i] > v2.v[i] ? v1.v[i] : v2.v[i];
		return v;
	}

	__device__
		T getValue(int i) {
		return v[i];
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
};
