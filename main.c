/*
 * Amro Qutna 1231229 sec 1
 * Bashar Ghosheh 1230368 sec 1
 *
 * Dr. Yazzan Abu Farha
 * AI project: Hospital location optimization
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define N 100       // Number of population points
#define M 100       // Number of candidate hospital locations
#define MAX_ITER 500

typedef struct {
    double x;
    double y;
} Point;

Point population[N];    // Coordinates of each population point
int   weights[N];       // Number of people at each population point
Point candidates[M];    // Coordinates of each candidate hospital site
double lambda;

// state[j] = 1 ->  hospital is built at site j
// state[j] = 0  ->  hospital is NOT built at site j
int current_state[M]; 
int best_state[M];   

// Snapshot to export map
// Captured at lambda = 50 to give a balanced geographic view (neither too many nor too few hospitals)
int snapshot_hc[M];
int snapshot_sa[M];

double calculate_total_cost(int state[M]) {
    double travel_cost = 0.0;
    int open_hospitals_count = 0;

    // 1. For each population point, find distance to nearest open hospital
    for (int i = 0; i < N; i++) {
        double min_dist = -1.0; // Start with -1 because no hospital found yet

        for (int j = 0; j < M; j++) {
            if (state[j] == 1) {
                // Euclidean distance between population point i and candidate j
                double dist = sqrt(pow(population[i].x - candidates[j].x, 2) + pow(population[i].y - candidates[j].y, 2));
                if (min_dist < 0 || dist < min_dist) {
                    min_dist = dist;
                }
            }
        }

        // return heavy penalty if no hospital is open (infeasible solution)
        if (min_dist < 0)
            return 9999999.0;

        travel_cost += weights[i] * min_dist;
    }

    // Building cost = λ × number of open hospitals
    for (int j = 0; j < M; j++) {
        if (state[j] == 1)
            open_hospitals_count++;
    }

    return travel_cost + (lambda * open_hospitals_count);
}

// This function is used to generate a random initial solution for both HC and SA.
// State is filled with a random binary solution where each site has 10-20% probability of being selected.
void initialize_random_state(int state[M]) {
    for (int j = 0; j < M; j++) {
        if(rand() % 100 < 15) {
            state[j] = 1; // Open hospital at site j
        } else {
            state[j] = 0; // Do not open hospital at site j
        }
    }
}

void hill_climbing() {
    initialize_random_state(current_state);
    double current_cost = calculate_total_cost(current_state);

    // Seed best_state with the initial solution
    memcpy(best_state, current_state, sizeof(int) * M);

    double best_cost = current_cost;

    for (int iter = 0; iter < MAX_ITER; iter++) {
        int i = rand() % M;

        // Flip bit to generate a neighbour
        current_state[i] = 1 - current_state[i]; // Flip 0 to 1 or 1 to 0
        double neighbor_cost = calculate_total_cost(current_state);

        if (neighbor_cost < current_cost) {
            // neighbour is better: accept it
            current_cost = neighbor_cost;

            // if this neighbor's cost is the best until now, update best_state
            if (current_cost < best_cost) {
                best_cost = current_cost;
                memcpy(best_state, current_state, sizeof(int) * M);
            }
        } else {
            // neighbour is worse: reject it and undo flip
            current_state[i] = 1 - current_state[i];
        }
    }

    // After HC finishes, best_state holds the best solution found. Copy it back to current_state.
    memcpy(current_state, best_state, sizeof(int) * M);
}


void simulated_annealing() {
    initialize_random_state(current_state);
    memcpy(best_state, current_state, sizeof(int) * M);

    double current_cost = calculate_total_cost(current_state);
    double best_cost    = current_cost;

    double T     = 1000.0;
    double alpha = 0.95; 

    // Run until temperature is very close to zero
    while (T > 0.00000000001) {
        int idx = rand() % M;

        // Flip one bit to generate a neighbour
        current_state[idx] = 1 - current_state[idx]; // Flip 0 to 1 or 1 to 0
        double neighbor_cost = calculate_total_cost(current_state);

        double delta_e = neighbor_cost - current_cost;

        int accept = 0;
        if (delta_e <= 0) {
            // Better or equal solution: Accept
            accept = 1;
        } else {
            // Worse solution: accept with probability
            double acceptance_probability = exp(-delta_e / T);
            if ((double)rand() / RAND_MAX < acceptance_probability) {
                accept = 1;
            }
        }

        if (accept) {
            current_cost = neighbor_cost;
            // if this neighbor's cost is the best until now, update best_state
            if (current_cost < best_cost) {
                best_cost = current_cost;
                memcpy(best_state, current_state, sizeof(int) * M);
            }
        } else {
            // neighbour is worse: reject it and undo flip
            current_state[idx] = 1 - current_state[idx];
        }

        T *= alpha; // Cool down
    }
}
    // Open output Files
    // We want to make the algorithm in C but we also want to visualise results in Python
    // So we export to CSV files that Python can easily read:
    // we will write three CSV files:
    // metrics.csv will contain run results for algorithms and all λ values
    // population.csv will contain the coordinates and weights of the population points
    // candidates.csv will contain the coordinates of candidate sites and whether they were selected in the HC and SA solutions at λ=50

int main() {
    srand(368);

    // 1. Populate initial benchmark coordinates over a uniform [0, 100] grid
    for (int i = 0; i < N; i++) {
        population[i].x = (double)(rand() % 101);
        population[i].y = (double)(rand() % 101);
        weights[i] = (rand() % 10) + 1; // Population densities assigned between 1 and 10
    }
    for (int j = 0; j < M; j++) {
        candidates[j].x = (double)(rand() % 101);
        candidates[j].y = (double)(rand() % 101);
    }

    // 2. Setup metric collection files for Python automation pipeline
    FILE *f_metrics = fopen("metrics.csv", "w");
    if (!f_metrics) {
        perror("Error creating metrics.csv");
        return 1;
    }
    fprintf(f_metrics, "Algorithm,Lambda,Run,Hospitals,TotalCost,Time\n");

    FILE *f_pop = fopen("population.csv", "w");
    if (!f_pop) {
        perror("Error creating population.csv");
        return 1;
    }
    fprintf(f_pop, "X,Y,Weight\n");
    for (int i = 0; i < N; i++)
        fprintf(f_pop, "%f,%f,%d\n", population[i].x, population[i].y, weights[i]);
    fclose(f_pop);

    // 3. Define log-scale Lambda (λ) test spectrum
    double lambda_values[4] = {1.0, 10.0, 50.0, 100.0};

    printf("==================================================================\n");
    printf("                  HOSPITAL LOCATION BENCHMARK SYSTEM              \n");
    printf("==================================================================\n");

    for (int l = 0; l < 4; l++) {
        lambda = lambda_values[l];

        double hc_sum_cost = 0.0, sa_sum_cost = 0.0;
        int    hc_sum_hospitals = 0, sa_sum_hospitals = 0;
        double hc_sum_time = 0.0,  sa_sum_time = 0.0;
        int    hc_wins = 0, sa_wins = 0, ties = 0;

        // Perform 10 independent experimental runs per configuration to track stability
        for (int run = 1; run <= 10; run++) {

            // --- Hill Climbing ---
            clock_t start1 = clock();
            hill_climbing();
            clock_t end1 = clock();
            double time1 = (double)(end1 - start1) / CLOCKS_PER_SEC;

            int hc_hospitals = 0;
            for (int j = 0; j < M; j++)
                if (best_state[j] == 1)
                    hc_hospitals++;

            double hc_cost = calculate_total_cost(best_state);

            if (lambda == 50.0)
                memcpy(snapshot_hc, best_state, sizeof(int) * M);

            fprintf(f_metrics, "HC,%f,%d,%d,%f,%f\n", lambda, run, hc_hospitals, hc_cost, time1);

            // --- Simulated Annealing ---
            clock_t start2 = clock();
            simulated_annealing();
            clock_t end2 = clock();
            double time2 = (double)(end2 - start2) / CLOCKS_PER_SEC;

            int sa_hospitals = 0;
            for (int j = 0; j < M; j++)
                if (best_state[j] == 1)
                    sa_hospitals++;

            double sa_cost = calculate_total_cost(best_state);

            if (lambda == 50.0)
                memcpy(snapshot_sa, best_state, sizeof(int) * M);

            fprintf(f_metrics, "SA,%f,%d,%d,%f,%f\n", lambda, run, sa_hospitals, sa_cost, time2);

            // Accumulate performance metrics
            hc_sum_cost += hc_cost;
            sa_sum_cost += sa_cost;
            hc_sum_hospitals += hc_hospitals;
            sa_sum_hospitals += sa_hospitals;
            hc_sum_time += time1;
            sa_sum_time += time2;

            if (hc_cost < sa_cost)
                hc_wins++;

            else if (sa_cost < hc_cost)
                sa_wins++;

            else
                ties++;
        }

        double hc_avg_cost = hc_sum_cost / 10.0;
        double sa_avg_cost = sa_sum_cost / 10.0;
        double hc_avg_time = hc_sum_time / 10.0;
        double sa_avg_time = sa_sum_time / 10.0;

        printf("\n--- Results for Lambda (λ) = %.1f (Averaged over 10 runs) ---\n", lambda);
        printf("  [Hill Climbing]\n");
        printf("    Avg Open Hospitals : %.1f\n",hc_sum_hospitals / 10.0);
        printf("    Avg Objective Cost : %.2f\n",hc_avg_cost);
        printf("    Avg Execution Time : %.4f seconds\n", hc_avg_time);

        printf("\n  [Simulated Annealing]\n");
        printf("    Avg Open Hospitals : %.1f\n",sa_sum_hospitals / 10.0);
        printf("    Avg Objective Cost : %.2f\n",sa_avg_cost);
        printf("    Avg Execution Time : %.4f seconds\n", sa_avg_time);

        printf("\n  [Algorithm Performance Comparison Summary]\n");
        printf("    Head-to-Head Record: SA Wins: %d | HC Wins: %d | Ties: %d\n", sa_wins, hc_wins, ties);
        printf("    Cost Improvement   : %.2f%%\n", ((hc_avg_cost - sa_avg_cost) / hc_avg_cost) * 100.0);
        printf("==================================================================\n");
    }

    fclose(f_metrics);

    // Save final spatial solution nodes mapping
    FILE *f_cand = fopen("candidates.csv", "w");
    if (!f_cand) {
        perror("Error creating candidates.csv");
        return 1;
    }
    fprintf(f_cand, "X,Y,HC_Open,SA_Open\n");
    for (int j = 0; j < M; j++) {
        fprintf(f_cand, "%f,%f,%d,%d\n", candidates[j].x, candidates[j].y, snapshot_hc[j], snapshot_sa[j]);
    }
    
    
    fclose(f_cand);

    printf("\n Data successfully exported to CSV files for Python visualization!\n");
    printf(" (candidates.csv reflects solutions at lambda = 50)\n\n");

    return 0;
}
