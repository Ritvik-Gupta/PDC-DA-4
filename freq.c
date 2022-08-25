#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <time.h>

#define DATASET_SIZE 1000000

void init_dataset(int dataset [], int* elm) {
	srand(time(NULL));

	for (int i = 0; i < DATASET_SIZE; ++i)
		dataset[i] = rand() % 100;

	*elm = rand() % 100;
}

int compute_frequency(int dataset [], int size, int elm) {
	int freq = 0;
	for (int i = 0; i < size; ++i)
		freq += dataset[i] == elm;
	return freq;
}

void main(int argc, char* argv []) {
	int rank, world_size;
	double start_stamp, end_stamp;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int* dataset;
	int freq_of_elm;
	if (rank == 0) {
		dataset = (int*)calloc(DATASET_SIZE, sizeof(int));
		init_dataset(dataset, &freq_of_elm);
	}

	// Start of Recording time
	start_stamp = MPI_Wtime();
	int elms_per_process = DATASET_SIZE / world_size;

	// Use MPI Collective Communication principles for Distributing work
	int* subset = (int*)calloc(elms_per_process, sizeof(int));
	// Scatter the Dataset to each process evenly
	MPI_Scatter(dataset, elms_per_process, MPI_INT, subset, elms_per_process, MPI_INT, 0, MPI_COMM_WORLD);
	// Also broadcast a copy of element to find in dataset
	MPI_Bcast(&freq_of_elm, 1, MPI_INT, 0, MPI_COMM_WORLD);

	int freq = compute_frequency(subset, elms_per_process, freq_of_elm);
	int total_freq = 0;
	MPI_Reduce(&freq, &total_freq, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	if (rank == 0) {
		end_stamp = MPI_Wtime();
		printf("Distributed Reduced Frequency of %d = %d\n\ttook %.15g\n", freq_of_elm, total_freq, end_stamp - start_stamp);

		start_stamp = MPI_Wtime();
		total_freq = compute_frequency(dataset, DATASET_SIZE, freq_of_elm);
		end_stamp = MPI_Wtime();
		printf("Sequential Frequency of          %d = %d\n\ttook %.15g\n", freq_of_elm, total_freq, end_stamp - start_stamp);
	}

	MPI_Finalize();
}
