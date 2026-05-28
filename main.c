#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N 100       // Number of population points
#define M 100       // Number of candidate hospital locations
#define MAX_ITER 500 // Max iterations for Hill Climbing / SA

// Structure to hold coordinate pairs
typedef struct {
    double x;
    double y;
} Point;

// Global problem instances
Point population[N];
int weights[N];    //number of people
Point candidates[M];
double lambda; // Controls hospital expense parameter (λ)

// State representations: arrays where index represents a candidate site,
// 1 means a hospital is open, and 0 means it's closed
int current_state[M];
int best_state[M];

// Objective Cost Function to minimize
double calculate_total_cost(int state[M]) {
    double travel_cost = 0.0;
    int open_hospitals_count = 0;

    // 1. Calculate travel cost for all people
    for (int i = 0; i < N; i++) {
        double min_dist = -1.0;

        for (int j = 0; j < M; j++) {
            if (state[j] == 1) { // If hospital candidate j is active
                // Euclidean distance formula
                double dist = sqrt(pow(population[i].x - candidates[j].x, 2) +
                                   pow(population[i].y - candidates[j].y, 2));

                if (min_dist < 0 || dist < min_dist) {
                    min_dist = dist;
                }
            }
        }

        // If no hospitals are selected open at all, return a heavily penalized cost
        if (min_dist < 0)
            return 9999999.0;

        // Accumulate weighted distance
        travel_cost += weights[i] * min_dist;
    }

    // 2. Count active hospitals and compute total building penalty
    for (int j = 0; j < M; j++) {
        if (state[j] == 1) {
            open_hospitals_count++;
        }
    }

    double building_cost = lambda * open_hospitals_count; // λ * |HC|
    return travel_cost + building_cost;
}

// Generates a random initial guess with roughly 10-20% hospitals open
void initialize_random_state(int state[M]) {
    for (int j = 0; j < M; j++) {
        if ((rand() % 100) < 15) { // ~15% chance to be selected
            state[j] = 1;
        }
        else {
            state[j] = 0;
        }
    }
}


// Algorithm 1: First-Choice Hill Climbing Variant
void hill_climbing() {
    initialize_random_state(current_state);
    double current_cost = calculate_total_cost(current_state);

    int local_minimum_reached = 0;
    int attempts = 0;

    // Run until termination threshold or max limit is reached
    while (!local_minimum_reached && attempts < MAX_ITER) {
        // Pick a neighbor by randomly mutating a single bit index
        int random_index = rand() % M;

        // 1. Make a random tweak (The Flip)
        current_state[random_index] = !current_state[random_index];
        // 2. Evaluate the new cost score
        double neighbor_cost = calculate_total_cost(current_state);
        // 3. Keep or reject
        if (neighbor_cost < current_cost) {
            current_cost = neighbor_cost; // It was a good change! Keep it.
            attempts = 0;
        } else {
            // It made the system worse! Revert it back to its original state
            current_state[random_index] = !current_state[random_index];
            attempts++;
        }

        // If tried 500 times consecutively without any drop in cost, stop
        if (attempts >= MAX_ITER) {
            local_minimum_reached = 1;
        }
    }
}

// Algorithm 2: Simulated Annealing Optimization Strategy
void simulated_annealing() {
    initialize_random_state(current_state);

    for (int j = 0; j < M; j++) {
        best_state[j] = current_state[j];
    }

    double current_cost = calculate_total_cost(current_state);
    double best_cost = current_cost;

    double T = 1000.0;       // Initial temperature T0 = 1000
    double alpha = 0.95;     // Cooling factor alpha = 0.95

    while (T > 0.0001){
        // Tweak one bit randomly
        int random_index = rand() % M;
        current_state[random_index] = !current_state[random_index];

        double neighbor_cost = calculate_total_cost(current_state);

        // Cost minimization formulation: Delta E = Current Cost - Neighbor Cost
        // Positive means neighbor is superior (lower cost score)
        double delta_e = current_cost - neighbor_cost;

        if (delta_e > 0) {
            // Better state option: accept update immediately
            current_cost = neighbor_cost;
            if (current_cost < best_cost) {
                best_cost = current_cost;
                for (int j = 0; j < M; j++) {
                    best_state[j] = current_state[j];
                }
            }
        } else {
            // Worse path option: calculate escape probability using energy variation
            double random_probability = (double)rand() / RAND_MAX;
            if (random_probability < exp(delta_e / T)) {
                current_cost = neighbor_cost; // Stochastically accept downhill path anyway
            } else {
                current_state[random_index] = !current_state[random_index]; // Return to original form
            }
        }

        T = T * alpha; // Slowly lower temperature scale
    }
}

int main() {
    // Seed random generation for dynamic profile parameters
    srand(68);//time(NULL)

    // Setup uniform baseline instance metrics over domain region [0,100]
    for(int i = 0; i < N; i++) {
        population[i].x = (rand() % 101);
        population[i].y = (rand() % 101);
        weights[i] = (rand() % 10) + 1; // Integer weights ranging uniformly from [1, 10]
    }
    for(int j = 0; j < M; j++) {
        candidates[j].x = (rand() % 101);
        candidates[j].y = (rand() % 101);
    }

    // Array of lambda values to evaluate log-scale experimental differences
    double lambda_values[4] = {1.0, 10.0,50.0, 100.0};

    printf("==================================================================\n");
    printf("                  HOSPITAL LOCATION BENCHMARK SYSTEM              \n");
    printf("==================================================================\n\n");

    for (int l = 0; l < 4; l++) {
        lambda = lambda_values[l];
        printf("--- Running Experiments for Lambda (λ) = %.1f ---\n", lambda);

        // 1. Profile Hill Climbing Implementation
        clock_t start_hc = clock();
        hill_climbing();
        clock_t end_hc = clock();
        double time_hc = (double)(end_hc - start_hc) / CLOCKS_PER_SEC;

        int hc_hospitals = 0;

        for(int j=0; j<M; j++)
            if(current_state[j] == 1)
                hc_hospitals++;

        printf("[Hill Climbing]\n");
        printf("   -> Execution Time: %f seconds\n", time_hc);
        printf("   -> Open Hospitals: %d\n", hc_hospitals);
        printf("   -> Evaluated Objective Cost: %f\n\n", calculate_total_cost(current_state));

        // 2. Profile Simulated Annealing Implementation
        clock_t start_sa = clock();
        simulated_annealing();
        clock_t end_sa = clock();
        double time_sa = (double)(end_sa - start_sa) / CLOCKS_PER_SEC;

        int sa_hospitals = 0;
        for(int j=0; j<M; j++)
            if(best_state[j] == 1)
                sa_hospitals++;

        printf("[Simulated Annealing]\n");
        printf("   -> Execution Time: %f seconds\n", time_sa);
        printf("   -> Open Hospitals: %d\n", sa_hospitals);
        printf("   -> Evaluated Objective Cost: %f\n", calculate_total_cost(best_state));
        printf("------------------------------------------------------------------\n\n");
    }

    return 0;
}