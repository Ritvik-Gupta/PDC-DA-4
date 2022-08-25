#include <stdio.h>
#include <mpi.h>
#include <time.h>
#include <stdlib.h>

#define DATASET_SIZE 10000000

void init_dataset(double dataset []) {
	srand(time(NULL));
	for (int i = 0; i < DATASET_SIZE; ++i)
		dataset[i] = (double)(rand() % 1000) / 1000.0;
}

double compute_average(double subset [], int size) {
	double total = 0.0;
	for (int i = 0; i < size; ++i)
		total += subset[i];
	return total / (double)size;
}

void main(int argc, char* argv []) {
	double* dataset;
	double start_stamp, end_stamp;
	int rank, world_size;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (rank == 0) {
		dataset = (double*)calloc(DATASET_SIZE, sizeof(double));
		init_dataset(dataset);
	}

	start_stamp = MPI_Wtime();
	double* subset = (double*)calloc(DATASET_SIZE, sizeof(double));

	int elms_per_process = DATASET_SIZE / world_size;
	MPI_Scatter(dataset, elms_per_process, MPI_DOUBLE, subset, elms_per_process, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	double sub_avg = compute_average(subset, elms_per_process);

	double* sub_averages;
	if (rank == 0) {
		sub_averages = (double*)calloc(world_size, sizeof(double));
	}

	MPI_Gather(&sub_avg, 1, MPI_DOUBLE, sub_averages, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if (rank == 0) {
		double overall_average = compute_average(sub_averages, world_size);
		end_stamp = MPI_Wtime();
		printf("Distributed Gathered Average = %.15g \n\ttook %.15g secs\n", overall_average, end_stamp - start_stamp);

		start_stamp = MPI_Wtime();
		overall_average = compute_average(dataset, DATASET_SIZE);
		end_stamp = MPI_Wtime();
		printf("Sequential Average           = %.15g \n\ttook %.15g secs\n", overall_average, end_stamp - start_stamp);
	}

	MPI_Finalize();
}
