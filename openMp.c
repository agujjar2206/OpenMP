#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <stdlib.h>
#include <omp.h>
#define MAX_LEN 12000000
#define NUM_THREADS 12

typedef struct {
    long maxProduct;
    long minProduct;
    int idx_min;
    int idx_max;
} result_t;

typedef struct {
    result_t *result;
    char pad[120];
    // Padding to enforce cacheline
} aligned_result_t;

void print_result(result_t *result){
    printf("sizeof result=%0ld\n", sizeof(result_t));
    printf("maxProduct = %0ld minProduct = %0ld Max index = %0d Min index = %0d", 
        result->maxProduct, result->minProduct, result->idx_max, result->idx_min);
    printf("\n");
}

result_t * create_n_init_result(void){
    result_t *result = NULL;
    result = (result_t *)malloc(sizeof(result_t));
    if(result == NULL) {
        printf("Out of Memory error for result\n");
        exit(1);
    }
    result->maxProduct = 0;
    result->minProduct = (1<<30) - 1;
    result->idx_max = 0;
    result->idx_min = 0;
    return result;
}

result_t * find_cluster_indexes_openMP(int *x, int *y, int *z)
{
    result_t *result, *per_thread_result;
    int i, id;
    int dist;
    aligned_result_t *aligned_result;
    
    result = create_n_init_result();
    
    aligned_result = (aligned_result_t *)malloc(sizeof(aligned_result_t) * NUM_THREADS);
    if(aligned_result == NULL) {
        printf("Out of Memory error for aligned result\n");
        exit(1);
    }
    for(int j = 0; j < NUM_THREADS; j++) {
        aligned_result[j].result = create_n_init_result();
    }
    

    #pragma omp parallel num_threads(NUM_THREADS) private(i, id, dist, per_thread_result) shared(aligned_result)
    {
        id = omp_get_thread_num();
        printf("Started thread id=%0d\n", id);
        per_thread_result = aligned_result[id].result;
        #pragma omp for
	    for(i = id ; i < MAX_LEN; i = i + NUM_THREADS){
		    dist = (x[i]) * (x[i]) + (y[i]) * (y[i]) + (z[i]) * (z[i]) ;
		if(dist > per_thread_result->maxProduct){
		    per_thread_result->maxProduct = dist;
		    per_thread_result->idx_max = i;
		}
		if(dist < aligned_result[id].result->minProduct){
		    per_thread_result->minProduct = dist;
		    per_thread_result->idx_min = i;
		}
	    }
        //printf("Finished thread id=%0d\n", id);
    }
    
    printf("Done with all threads\n");
    for(int j = 0; j < NUM_THREADS; j++){
        if(aligned_result[j].result->maxProduct > result->maxProduct){
            result->maxProduct = aligned_result[j].result->maxProduct;
            result->idx_max = aligned_result[j].result->idx_max;
        }
        if(aligned_result[j].result->minProduct < result->minProduct){
            result->minProduct = aligned_result[j].result->minProduct;
            result->idx_min = aligned_result[j].result->idx_min;
        }
    }
    free(aligned_result);
    return result;
}

result_t * find_cluster_indexes(int *x, int *y, int *z)
{
    int dist;
    result_t *result;
    result = create_n_init_result();
    // Initialize with a big value
    for(int i=0; i < MAX_LEN; i++) {
        dist = (x[i]) * (x[i]) + (y[i]) * (y[i]) + (z[i]) * (z[i]) ;
        if(dist > result->maxProduct){
            result->maxProduct = dist;
            result->idx_max = i;
        }
        if(dist < result->minProduct){
            result->minProduct = dist;
            result->idx_min = i;
        }
    }
    
    return result;
}

int main()
{
    int *x, *y, *z;
    result_t *result_normal, *result_openmp;
    x = (int *) malloc(sizeof(int) * MAX_LEN);
    if(x == NULL) {
        printf("Out of memory in Malloc for x");
        return 1;
    }
    y = (int *) malloc(sizeof(int) * MAX_LEN);
    if(y == NULL) {
        printf("Out of memory in Malloc for y");
        return 1;
    }
    z = (int *) malloc(sizeof(int) * MAX_LEN);
    if(z == NULL) {
        printf("Out of memory in Malloc for z");
        return 1;
    }
    // Fill some random data
    for (int i = 0; i < MAX_LEN; i++){
        x[i] = rand() % (1<<10);
        y[i] = rand() % (1<<10);
        z[i] = rand() % (1<<10);
    }
    printf("Done random value initialization\n");
    double time_spent1 = 0.0;
    double time_spent2 = 0.0;
    /////////////
    clock_t begin = clock();
    result_normal = find_cluster_indexes(x, y, z);
    clock_t end = clock();
    printf("normal find_cluster_indexes done\n");
    //print_result(result_normal);
    free(result_normal);
    time_spent1 += (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Time elpased is %f seconds for naive cluster finding.\n\n", time_spent1);
    ///////////////
    begin = clock();
    result_openmp = find_cluster_indexes_openMP(x, y, z);
    end = clock();
    printf("openMP find_cluster_indexes_openMP done\n");
    //print_result(result_openmp);
    free(result_openmp);
    time_spent2 += (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Time elpased is %f seconds for OpenMP cluster finding \n", time_spent2);
    printf("%f X time faster.\n", time_spent1/time_spent2);
    // Free the memory
    free(x);
    free(y);
    free(z);
    //printf("TEST=%0lld\n", sum_vectorized_256(x));
    return 0;
}
