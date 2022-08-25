#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <time.h>

#define DATASET_SIZE 100000000

void init_dataset(double dataset []) {
	srand(time(NULL));
	for (int i = 0; i < DATASET_SIZE; ++i)
		dataset[i] = (double)(rand() % 1000) / 1000.0;
}

double compute_average(double dataset [], int size) {
	double total = 0.0;
	for (int i = 0; i < size; ++i)
		total += dataset[i];
	return total / (double)size;
}

void main(int argc, char* argv []) {
	int rank, world_size;
	double start_stamp, end_stamp;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	double* dataset;
	if (rank == 0) {
		dataset = (double*)calloc(DATASET_SIZE, sizeof(double));
		init_dataset(dataset);
	}

	start_stamp = MPI_Wtime();

	int elms_per_process = DATASET_SIZE / world_size;
	double* subset = (double*)calloc(elms_per_process, sizeof(double));
	MPI_Scatter(dataset, elms_per_process, MPI_DOUBLE, subset, elms_per_process, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	double sub_average = compute_average(subset, elms_per_process);

	double overall_average = 0.0;
	MPI_Reduce(&sub_average, &overall_average, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	overall_average /= world_size;

	if (rank == 0) {
		end_stamp = MPI_Wtime();
		printf("Distributed Reduced Average = %.15g\n\ttook %.15g secs\n", overall_average, end_stamp - start_stamp);

		start_stamp = MPI_Wtime();
		overall_average = compute_average(dataset, DATASET_SIZE);
		end_stamp = MPI_Wtime();
		printf("Sequential Average          = %.15g\n\ttook %.15g secs\n", overall_average, end_stamp - start_stamp);
	}

	MPI_Finalize();
}

