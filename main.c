#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define N 100            // Number of population points (people needing coverage)
#define M 100            // Number of candidate hospital locations
#define HC_MAX_ITER 500  // Maximum iterations for the Hill Climbing baseline

// Readability Constants for Binary States
#define HOSPITAL_OPEN   1
#define HOSPITAL_CLOSED 0

// Structure to hold a 2D geographic coordinate pair
typedef struct {
    double x;
    double y;
} Point;

// Global Data Structures
Point population[N];   // Coordinates of each population point
int   weights[N];      // Population weights (number of people at each point)
Point candidates[M];   // Coordinates of potential hospital sites
double lambda;         // Penalty multiplier (λ) controlling build vs. travel tradeoff

// State representations: Binary arrays where 1 = Open Hospital, 0 = Closed
int current_state[M];  // State currently being explored by the search step
int best_state[M];     // Global best state found during a single multi-run trial

// Snapshot memory arrays to hold optimal solution mapping for λ = 50 visualization
int snapshot_hc[M];
int snapshot_sa[M];

/**
 * OBJECTIVE/COST FUNCTION
 * Evaluates the absolute fitness landscape metric for a given binary state configuration.
 * Cost = Sum(weight_i * distance_to_closest_hospital) + (λ * number_of_open_hospitals)
 */
double calculate_total_cost(int state[M]) {
    double travel_cost = 0.0;
    int open_hospitals_count = 0;

    // 1. Compute patient accessibility overhead (Weighted Euclidean Travel Distance)
    for (int i = 0; i < N; i++) {
        double min_dist = -1.0;

        for (int j = 0; j < M; j++) {
            if (state[j] == HOSPITAL_OPEN) { // Only check distance to active/open hospitals
                double dist = sqrt(
                    pow(population[i].x - candidates[j].x, 2) +
                    pow(population[i].y - candidates[j].y, 2)
                );
                if (min_dist < 0 || dist < min_dist) {
                    min_dist = dist;
                }
            }
        }

        // Hard Infeasibility Guard: If zero hospitals are open, return a massive penalty
        if (min_dist < 0) {
            return 9999999.0;
        }

        travel_cost += weights[i] * min_dist; // Contribution scales by population size
    }

    // 2. Compute construction budget cost (λ * |HC|)
    for (int j = 0; j < M; j++) {
        if (state[j] == HOSPITAL_OPEN) {
            open_hospitals_count++;
        }
    }

    return travel_cost + (lambda * open_hospitals_count);
}

/**
 * RANDOM INITIALIZATION
 * Randomly opens ~15% of candidate locations to seed the local search space.
 * Replaced the ternary conditional operator with an explicit if-else branch structure.
 */
void initialize_random_state(int state[M]) {
    for (int j = 0; j < M; j++) {
        int roll = rand() % 100;
        if (roll < 15) {
            state[j] = HOSPITAL_OPEN;
        } else {
            state[j] = HOSPITAL_CLOSED;
        }
    }
}

/**
 * GREEDY HILL CLIMBING ALGORITHM
 * Explores immediate neighborhood states by flipping a single candidate bit.
 * Uses logical negation (!) instead of bitwise XOR (^) to flip states.
 */
void hill_climbing() {
    // Generate an initial random solution setup
    initialize_random_state(current_state);
    double current_cost = calculate_total_cost(current_state);

    // Seed the best tracker with this start position
    memcpy(best_state, current_state, sizeof(int) * M);
    double best_cost = current_cost;

    for (int iter = 0; iter < HC_MAX_ITER; iter++) {
        // Pick a random candidate site to flip (Neighborhood operator)
        int idx = rand() % M;

        // FLIPPING STATE BIT: Changed from ^= 1 to logical negation (!)
        current_state[idx] = !current_state[idx];
        double neighbor_cost = calculate_total_cost(current_state);

        // GREEDY EVALUATION: If the neighbor decreases total cost, accept the transition
        if (neighbor_cost < current_cost) {
            current_cost = neighbor_cost;

            // Log if this is the absolute best configuration seen across iterations
            if (current_cost < best_cost) {
                best_cost = current_cost;
                memcpy(best_state, current_state, sizeof(int) * M);
            }
        } else {
            // REJECTION LOGIC: If the move worsens cost, undo the change with another logical negation
            current_state[idx] = !current_state[idx];
        }
    }

    // Force the final working buffer to reflect the optimal found path setup
    memcpy(current_state, best_state, sizeof(int) * M);
}

/**
 * SIMULATED ANNEALING ALGORITHM
 * Uses a stochastic cooling process to escape local optima.
 * Uses logical negation (!) instead of bitwise XOR (^) to explore neighbor states.
 */
void simulated_annealing() {
    initialize_random_state(current_state);
    memcpy(best_state, current_state, sizeof(int) * M);

    double current_cost = calculate_total_cost(current_state);
    double best_cost    = current_cost;

    double T     = 1000.0; // Starting Temperature (High exploration)
    double alpha = 0.95;   // Cooling Rate multiplier

    // Loop continues exploring until system temperature converges close to zero
    while (T > 0.00000000001) {
        int idx = rand() % M;

        // FLIPPING STATE BIT
        current_state[idx] = !current_state[idx];
        double neighbor_cost = calculate_total_cost(current_state);

        // Delta E: (New Cost - Old Cost). Negative means an improvement.
        double delta_e = neighbor_cost - current_cost;

        int accept = 0;
        if (delta_e <= 0) {
            // Improvement or equal cost: Always accept the transition
            accept = 1;
        } else {
            // Solution degraded: Accept stochastically based on Boltzmann probability
            double acceptance_probability = exp(-delta_e / T);
            double roll = (double)rand() / RAND_MAX;
            if (roll < acceptance_probability) {
                accept = 1;
            }
        }

        if (accept) {
            current_cost = neighbor_cost;
            // Track absolute minimum achieved
            if (current_cost < best_cost) {
                best_cost = current_cost;
                memcpy(best_state, current_state, sizeof(int) * M);
            }
        } else {
            // REJECTION LOGIC: Move denied, flip bit back using logical negation
            current_state[idx] = !current_state[idx];
        }

        // Apply geometric cooling schedule parameter
        T *= alpha;
    }
}

int main() {
    // Fixed Random Seed (Guarantees reproducible coordinates and matrix data)
    // NOTE: To make instances fully randomized per execution, swap to: srand(time(NULL));
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
    if (!f_metrics) { perror("Error creating metrics.csv"); return 1; }
    fprintf(f_metrics, "Algorithm,Lambda,Run,Hospitals,TotalCost,Time\n");

    FILE *f_pop = fopen("population.csv", "w");
    if (!f_pop) { perror("Error creating population.csv"); return 1; }
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

            // --- Test Variant A: Hill Climbing ---
            clock_t start1 = clock();
            hill_climbing();
            clock_t end1 = clock();
            double time1 = (double)(end1 - start1) / CLOCKS_PER_SEC;

            int hc_hospitals = 0;
            for (int j = 0; j < M; j++)
                if (best_state[j] == HOSPITAL_OPEN) hc_hospitals++;
            double hc_cost = calculate_total_cost(best_state);

            if (lambda == 50.0)
                memcpy(snapshot_hc, best_state, sizeof(int) * M);

            fprintf(f_metrics, "HC,%f,%d,%d,%f,%f\n", lambda, run, hc_hospitals, hc_cost, time1);

            // --- Test Variant B: Simulated Annealing ---
            clock_t start2 = clock();
            simulated_annealing();
            clock_t end2 = clock();
            double time2 = (double)(end2 - start2) / CLOCKS_PER_SEC;

            int sa_hospitals = 0;
            for (int j = 0; j < M; j++)
                if (best_state[j] == HOSPITAL_OPEN) sa_hospitals++;
            double sa_cost = calculate_total_cost(best_state);

            if (lambda == 50.0)
                memcpy(snapshot_sa, best_state, sizeof(int) * M);

            fprintf(f_metrics, "SA,%f,%d,%d,%f,%f\n", lambda, run, sa_hospitals, sa_cost, time2);

            // Accumulate performance metrics
            hc_sum_cost      += hc_cost;
            sa_sum_cost      += sa_cost;
            hc_sum_hospitals += hc_hospitals;
            sa_sum_hospitals += sa_hospitals;
            hc_sum_time      += time1;
            sa_sum_time      += time2;

            if      (hc_cost < sa_cost) hc_wins++;
            else if (sa_cost < hc_cost) sa_wins++;
            else                        ties++;
        }

        // ==================================================================
        // ENHANCED TERMINAL REPORTING BLOCK (FOR REPORT TABLES)
        // ==================================================================
        double hc_avg_cost = hc_sum_cost / 10.0;
        double sa_avg_cost = sa_sum_cost / 10.0;
        double hc_avg_time = hc_sum_time / 10.0;
        double sa_avg_time = sa_sum_time / 10.0;

        printf("\n--- Results for Lambda (λ) = %.1f (Averaged over 10 runs) ---\n", lambda);
        printf("  [Hill Climbing]\n");
        printf("    Avg Open Hospitals : %.1f\n",        hc_sum_hospitals / 10.0);
        printf("    Avg Objective Cost : %.2f\n",        hc_avg_cost);
        printf("    Avg Execution Time : %.4f seconds\n", hc_avg_time);

        printf("\n  [Simulated Annealing]\n");
        printf("    Avg Open Hospitals : %.1f\n",        sa_sum_hospitals / 10.0);
        printf("    Avg Objective Cost : %.2f\n",        sa_avg_cost);
        printf("    Avg Execution Time : %.4f seconds\n", sa_avg_time);

        printf("\n  [Algorithm Performance Comparison Summary]\n");
        printf("    Head-to-Head Record: SA Wins: %d | HC Wins: %d | Ties: %d\n", sa_wins, hc_wins, ties);
        printf("    Cost Improvement   : %.2f%%\n", ((hc_avg_cost - sa_avg_cost) / hc_avg_cost) * 100.0);
        printf("==================================================================\n");
    }

    fclose(f_metrics);

    // Save final spatial solution nodes mapping
    FILE *f_cand = fopen("candidates.csv", "w");
    if (!f_cand) { perror("Error creating candidates.csv"); return 1; }
    fprintf(f_cand, "X,Y,HC_Open,SA_Open\n");
    for (int j = 0; j < M; j++) {
        fprintf(f_cand, "%f,%f,%d,%d\n", candidates[j].x, candidates[j].y, snapshot_hc[j], snapshot_sa[j]);
    }
    fclose(f_cand);

    printf("\n Data successfully exported to CSV files for Python visualization!\n");
    printf(" (candidates.csv reflects solutions at lambda = 50)\n\n");

    return 0;
}