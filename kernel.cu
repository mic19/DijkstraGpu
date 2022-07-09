
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <numeric>
#include <queue>

const int MAX_THREADS_PER_BLOCK = 1024;

void print(const std::vector<int>& vec) {
	for (const auto& item : vec) {
		std::cout << item << ", ";
	}
	std::cout << std::endl;
}

__global__ void relax(int* distances, int* frontiers, int* successors, int* successors_weights, int size) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (thread_idx < size) {
		int frontier = frontiers[thread_idx];
		int successor = successors[thread_idx];
		int weight = successors_weights[thread_idx];
		//printf("From block_idx: %d thread_idx: %d, frontier: %d, successor: %d, weight: %d\n", blockIdx.x, thread_idx, frontier, successor, weight);
		//printf("From block idx: %d block idy: %d, blockDim x: %d, blockDim y: %d\n", blockIdx.x, blockIdx.y, blockDim.x, blockDim.y);
		atomicMin(distances + successor, distances[frontier] + weight);
	}
}

void getUnvisitedSuccessors(
	const int vert, const int* vertices, const int* edges, const int* weights, const bool* visited,
	std::vector<int>& successors, std::vector<int>& successors_weights, std::vector<int>& thread_frontiers) {
	
	for (int i = vertices[vert]; i < vertices[vert + 1]; ++i) {
		if (!visited[edges[i]]) {
			successors.push_back(edges[i]);
			successors_weights.push_back(weights[i]);
			thread_frontiers.push_back(vert);
		}
	}
}

std::vector<int> gpuDijkstra(
	const int* vertices, const int* edges, const int* weights, const int num_of_vertices, const int num_of_edges) {
	std::vector<int> frontiers{};
	frontiers.push_back(0);
	bool* visited = new bool[num_of_vertices]();
	visited[0] = true;
	std::vector<int> distances(num_of_vertices, INT_MAX);
	distances[0] = 0;
	
	int* d_frontiers;
	int* d_successors;
	int* d_successors_weights;
	cudaMalloc(&d_frontiers, sizeof(int) * 100 * num_of_vertices); // todo size
	cudaMalloc(&d_successors, sizeof(int) * 100 * num_of_vertices);
	cudaMalloc(&d_successors_weights, sizeof(int) * 100 * num_of_vertices);

	int* d_distances;
	cudaMalloc(&d_distances, sizeof(int) * num_of_vertices);
	cudaMemcpy(d_distances, distances.data(), sizeof(int) * num_of_vertices, cudaMemcpyHostToDevice);

	std::vector<int> successors{};
	std::vector<int> successors_weights{};
	std::vector<int> thread_frontiers{};

	int mssp = 0;
	while (mssp != INT_MAX) {
		// find unvisited successors of frontiers
		for (const auto vert : frontiers) {
			getUnvisitedSuccessors(vert, vertices, edges, weights, visited, successors, successors_weights, thread_frontiers);
		}

		// todo streams
		cudaError_t cudaStatus = cudaMemcpy(d_frontiers, thread_frontiers.data(), sizeof(int) * thread_frontiers.size(), cudaMemcpyHostToDevice);
		cudaMemcpy(d_successors, successors.data(), sizeof(int) * successors.size(), cudaMemcpyHostToDevice);
		cudaMemcpy(d_successors_weights, successors_weights.data(), sizeof(int) * successors_weights.size(), cudaMemcpyHostToDevice);
		
		if (!successors.empty()) {
			int total_threads = successors.size();
			if (total_threads <= MAX_THREADS_PER_BLOCK) {
				relax << <1, total_threads >> > (d_distances, d_frontiers, d_successors, d_successors_weights, total_threads);
			}
			else {
				relax << <total_threads / MAX_THREADS_PER_BLOCK + 1, MAX_THREADS_PER_BLOCK >> > (d_distances, d_frontiers, d_successors, d_successors_weights, total_threads);
			}
		}

		frontiers.clear();
		successors.clear();
		successors_weights.clear();
		thread_frontiers.clear();

		cudaDeviceSynchronize();
		cudaMemcpy(distances.data(), d_distances, sizeof(int) * num_of_vertices, cudaMemcpyDeviceToHost);

		mssp = INT_MAX;
		for (int i = 0; i < num_of_vertices; ++i) {
			if (visited[i] == false && mssp > distances[i]) {
				mssp = distances[i];
			}
		}

		for (int i = 0; i < num_of_vertices; ++i) {
			if (distances[i] == mssp) {
				frontiers.push_back(i);
				visited[i] = true;
			}
		}
	}

	cudaDeviceSynchronize();
	cudaFree(d_frontiers);
	cudaFree(d_successors);
	cudaFree(d_successors_weights);
	delete[] visited;

	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Program failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
	}

	return distances;
}

void generateRandomGraph(
	std::vector<int>& vertices, std::vector<int>& edges, std::vector<int>& weights, const int num_of_vertices, const int seed, const float density = 0.1) {
	std::vector<std::vector<std::pair<int, int>>> matrix(num_of_vertices, std::vector<std::pair<int, int>>{});

	std::mt19937 gen(seed);
	std::uniform_int_distribution<> distr(0, INT_MAX);
	std::uniform_int_distribution<> distr_weights(1, 10);

	for (int i = 0; i < num_of_vertices - 1; ++i) {
		for (int j = i + 1; j < num_of_vertices; ++j) {
			auto r = distr(gen);
			float p = static_cast<float>(distr(gen)) / INT_MAX;
			if (p < density) {
				int weight = distr_weights(gen);
				matrix[i].push_back({ j, weight });
				matrix[j].push_back({ i, weight });
			}
		}
	}

	vertices.reserve(num_of_vertices + 1);
	vertices.push_back(0);
	int end = 0;

	for (const auto& items : matrix) {
		vertices.push_back(end + items.size());
		end += items.size();
		std::for_each(items.begin(), items.end(), [&edges, &weights, &distr_weights, &gen](std::pair<int, int> item) {
			edges.push_back(item.first);
			weights.push_back(item.second);
			});
	}
}

//CPU
void getSuccessors(
	const int vert, const int* vertices, const int* edges, const int* weights,
	std::vector<int>& successors, std::vector<int>& successors_weights) {
	successors.clear();
	successors_weights.clear();

	for (int i = vertices[vert]; i < vertices[vert + 1]; ++i) {
		successors.push_back(edges[i]);
		successors_weights.push_back(weights[i]);
	}
}

std::vector<int> cpuDijkstra(
	const int* vertices, const int* edges, const int* weights, const int num_of_vertices, const int num_of_edges) {
	std::vector<int> distances(num_of_vertices, INT_MAX);
	distances[0] = 0;
	int frontier = 0;

	typedef std::pair<int, int> PqItem;
	std::priority_queue<PqItem, std::vector<PqItem>, std::greater<PqItem>> pq{};
	pq.push({ 0, 0 });

	std::vector<int> successors, successors_weights;
	successors.reserve(2 * num_of_edges / num_of_vertices);
	successors_weights.reserve(2 * num_of_edges / num_of_vertices);

	while (!pq.empty()) {
		getSuccessors(frontier, vertices, edges, weights, successors, successors_weights);

		for (int i = 0; i < successors.size(); ++i) {
			const int successor = successors[i];
			int dist = distances[frontier] + successors_weights[i];
			if (distances[successor] > dist) {
				distances[successor] = dist;
				pq.push({ distances[successor], successor });
			}
		}

		frontier = pq.top().second;
		pq.pop();
	}
	return distances;
}

int main() {
	const int num_of_vertices = 10000;

	//std::vector<int> std_vertices{ 0, 2, 5, 9, 11, 13, 15, 19, 20 };
	//std::vector<int> std_edges{ /*0*/ 1, 6, /*1*/ 0, 2, 4, /*2*/ 1, 3, 4, 6, /*3*/ 2, 5,
	///*4*/ 1, 2, /*5*/ 3, 6, /*6*/ 0, 2, 5, 7, /*7*/ 6 };
	//std::vector<int> std_weights{ /*0*/ 1, 2, /*1*/ 1, 2, 5, /*2*/ 2, 1, 2, 4, /*3*/ 1, 1,
	///*4*/ 5, 2, /*5*/ 1, 3, /*6*/ 2, 4, 3, 3, /*7*/ 3 };

	int tries = 10;
	int offset = 400;
	std::vector<bool> results(tries, true);
	std::vector<int> cpu_times;
	std::vector<int> gpu_times;

	for (int i = offset; i < offset + tries; ++i) {
		std::cout << "---------------------------------------------" << std::endl;
		std::random_device rd;

		std::vector<int> vertices;
		std::vector<int> edges;
		std::vector<int> weights;
		
		generateRandomGraph(vertices, edges, weights, num_of_vertices, i, 0.01);

		std::chrono::steady_clock::time_point begin_cpu = std::chrono::steady_clock::now();
		const auto cpu_distances = cpuDijkstra(vertices.data(), edges.data(), weights.data(), num_of_vertices, edges.size());
		std::chrono::steady_clock::time_point end_cpu = std::chrono::steady_clock::now();

		//std::cout << "------ CPU ------" << std::endl;
		//print(cpu_distances);

		std::chrono::steady_clock::time_point begin_gpu = std::chrono::steady_clock::now();
		const auto gpu_distances = gpuDijkstra(vertices.data(), edges.data(), weights.data(), num_of_vertices, edges.size());
		std::chrono::steady_clock::time_point end_gpu = std::chrono::steady_clock::now();
		//std::cout << "------ GPU ------" << std::endl;
		//print(gpu_distances);

		int cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - begin_cpu).count();
		int gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - begin_gpu).count();
		cpu_times.push_back(cpu_time);
		gpu_times.push_back(gpu_time);
		std::cout << "CPU execution time: " << cpu_time << " microseconds" << std::endl;
		std::cout << "GPU execution time: " << gpu_time << " microseconds" << std::endl;

		bool match = true;
		for (int j = 0; j < num_of_vertices; ++j) {
			if (cpu_distances[j] != gpu_distances[j]) {
				match = false;
				results[i - offset] = false;
			}
		}
	}

	for (int i = 0; i < tries; ++i) {
		std::cout << "seed: " << i + offset << " match: " << results[i] << std::endl;
	}

	int avg_cpu_time = std::accumulate(cpu_times.begin(), cpu_times.end(), 0) / tries;
	int avg_gpu_time = std::accumulate(gpu_times.begin(), gpu_times.end(), 0) / tries;

	std::cout << "Avg cpu time: " << avg_cpu_time << " Avg gpu time: " << avg_gpu_time << std::endl;
	std::cout << "GPU/CPU: " << static_cast<float>(avg_gpu_time) / avg_cpu_time << std::endl;

    return 0;
}
