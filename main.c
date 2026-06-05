#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define N 100        // Number of population points
#define M 100        // Number of candidate hospital locations
#define HC_MAX_ITER 500 // Max iterations for Hill Climbing

// Structure to hold a 2D coordinate pair
typedef struct {
    double x;
    double y;
} Point;

// ─── Global Problem Instance ──────────────────────────────────────────────────
Point population[N];   // Coordinates of each population point
int   weights[N];      // Number of people at each population point
Point candidates[M];   // Coordinates of each candidate hospital site
double lambda;         // Hospital cost parameter λ — controls build-vs-travel tradeoff

// ─── State Representations ────────────────────────────────────────────────────
// Each array has M entries; index j corresponds to candidate site j.
// state[j] = 1  →  hospital is built at site j
// state[j] = 0  →  site j is unused
int current_state[M]; // Working state (modified in-place during search)
int best_state[M];    // Best state found so far (used by both HC and SA)

// ─── Snapshot for Map Export ─────────────────────────────────────────────────
// Captured at lambda = 50 so the geographic plot is meaningful
int snapshot_hc[M];
int snapshot_sa[M];

// ─────────────────────────────────────────────────────────────────────────────
// calculate_total_cost
//
// Input : state[M] — a candidate solution (0/1 array of open hospitals)
// Output: the scalar objective value to MINIMISE:
//           cost = Σ_i ( w_i * min_j∈HC dist(P_i, C_j) )  +  λ * |HC|
//         Returns 9999999.0 if no hospital is open (infeasible penalty).
// ─────────────────────────────────────────────────────────────────────────────
double calculate_total_cost(int state[M]) {
    double travel_cost = 0.0;
    int open_hospitals_count = 0;

    // 1. For each population point, find distance to nearest open hospital
    for (int i = 0; i < N; i++) {
        double min_dist = -1.0;

        for (int j = 0; j < M; j++) {
            if (state[j] == 1) {
                // Euclidean distance between population point i and candidate j
                double dist = sqrt(
                    pow(population[i].x - candidates[j].x, 2) +
                    pow(population[i].y - candidates[j].y, 2)
                );
                if (min_dist < 0 || dist < min_dist) {
                    min_dist = dist;
                }
            }
        }

        // Infeasibility guard: no hospital open → heavily penalise
        if (min_dist < 0)
            return 9999999.0;

        travel_cost += weights[i] * min_dist; // weighted distance contribution
    }

    // 2. Building cost: λ × number of open hospitals
    for (int j = 0; j < M; j++) {
        if (state[j] == 1) open_hospitals_count++;
    }

    return travel_cost + lambda * open_hospitals_count;
}

// ─────────────────────────────────────────────────────────────────────────────
// initialize_random_state
//
// Input : state[M] — array to initialise
// Output: state is filled with a random binary solution where each site has
//         ~15% probability of being selected (gives roughly 10–20% open).
// ─────────────────────────────────────────────────────────────────────────────

void initialize_random_state(int state[M]) {
    for (int j = 0; j < M; j++) {
        state[j] = ((rand() % 100) < 15) ? 1 : 0;
    }
}


void hill_climbing() {
    // Initialise with a random solution
    initialize_random_state(current_state);
    double current_cost = calculate_total_cost(current_state);

    // Seed best_state with the initial solution
    memcpy(best_state, current_state, sizeof(int) * M);
    double best_cost = current_cost;

    for (int iter = 0; iter < HC_MAX_ITER; iter++) {
        int idx = rand() % M;

        // Flip candidate bit to generate a neighbour
        current_state[idx] ^= 1;
        double neighbor_cost = calculate_total_cost(current_state);

        if (neighbor_cost < current_cost) {
            // Accept improvement
            current_cost = neighbor_cost;

            // Update best if this is the global best seen so far
            if (current_cost < best_cost) {
                best_cost = current_cost;
                memcpy(best_state, current_state, sizeof(int) * M);
            }
        } else {
            // Reject — undo the flip
            current_state[idx] ^= 1;
        }
    }

    // Ensure current_state reflects the best solution found
    memcpy(current_state, best_state, sizeof(int) * M);
}


void simulated_annealing() {
    // Initialise with a random solution
    initialize_random_state(current_state);
    memcpy(best_state, current_state, sizeof(int) * M);

    double current_cost = calculate_total_cost(current_state);
    double best_cost    = current_cost;

    double T     = 1000.0; // Initial temperature
    double alpha = 0.95;  // Cooling rate (slow cooling → ~16,000 iterations,
                           // giving SA enough time to escape local minima)

    // Run until temperature effectively reaches zero
    while (T > 0.00000000001) {
        int idx = rand() % M;

        // Flip one bit to generate a neighbour
        current_state[idx] ^= 1;
        double neighbor_cost = calculate_total_cost(current_state);

        // Δe < 0 → improvement; Δe > 0 → degradation
        double delta_e = neighbor_cost - current_cost;

        int accept = 0;
        if (delta_e <= 0) {
            // Better or equal solution: always accept
            accept = 1;
        } else {
            // Worse solution: accept with Boltzmann probability
            double acceptance_probability = exp(-delta_e / T);
            if ((double)rand() / RAND_MAX < acceptance_probability) {
                accept = 1;
            }
        }

        if (accept) {
            current_cost = neighbor_cost;
            // Track global best
            if (current_cost < best_cost) {
                best_cost = current_cost;
                memcpy(best_state, current_state, sizeof(int) * M);
            }
        } else {
            // Reject move — undo flip
            current_state[idx] ^= 1;
        }

        T *= alpha; // Cool down
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// main
//
// 1. Generates a random problem instance (population + candidates).
// 2. Runs HC and SA 10 times for each λ ∈ {1, 10, 50, 100}.
// 3. Prints averaged summary statistics to stdout.
// 4. Writes detailed per-run metrics to metrics.csv.
// 5. Writes population data to population.csv.
// 6. Writes candidate site selections (at λ=50) to candidates.csv for the
//    geographic visualisation in main.py.
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    srand(368);

    // ── Generate problem instance over [0, 100] × [0, 100] ──────────────────
    for (int i = 0; i < N; i++) {
        population[i].x = (double)(rand() % 101);
        population[i].y = (double)(rand() % 101);
        weights[i] = (rand() % 10) + 1; // Uniform integer weight in [1, 10]
    }
    for (int j = 0; j < M; j++) {
        candidates[j].x = (double)(rand() % 101);
        candidates[j].y = (double)(rand() % 101);
    }

    // ── Open output files ────────────────────────────────────────────────────
    FILE *f_metrics = fopen("metrics.csv", "w");
    if (!f_metrics) { perror("Error creating metrics.csv"); return 1; }
    fprintf(f_metrics, "Algorithm,Lambda,Run,Hospitals,TotalCost,Time\n");

    FILE *f_pop = fopen("population.csv", "w");
    if (!f_pop) { perror("Error creating population.csv"); return 1; }
    fprintf(f_pop, "X,Y,Weight\n");
    for (int i = 0; i < N; i++)
        fprintf(f_pop, "%f,%f,%d\n", population[i].x, population[i].y, weights[i]);
    fclose(f_pop);

    // ── Lambda values (log-scale) ────────────────────────────────────────────
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

        for (int run = 1; run <= 10; run++) {

            // ── Hill Climbing ─────────────────────────────────────────────
            clock_t start1 = clock();
            hill_climbing();
            clock_t end1 = clock();
            double time1 = (double)(end1 - start1) / CLOCKS_PER_SEC;

            // HC result lives in best_state (and mirrored to current_state)
            int hc_hospitals = 0;
            for (int j = 0; j < M; j++)
                if (best_state[j] == 1) hc_hospitals++;
            double hc_cost = calculate_total_cost(best_state);

            // Save a snapshot of HC result at λ = 50 for the map plot
            if (lambda == 50.0)
                memcpy(snapshot_hc, best_state, sizeof(int) * M);

            fprintf(f_metrics, "HC,%f,%d,%d,%f,%f\n",
                    lambda, run, hc_hospitals, hc_cost, time1);

            // ── Simulated Annealing ───────────────────────────────────────
            clock_t start2 = clock();
            simulated_annealing();
            clock_t end2 = clock();
            double time2 = (double)(end2 - start2) / CLOCKS_PER_SEC;

            // SA result lives in best_state
            int sa_hospitals = 0;
            for (int j = 0; j < M; j++)
                if (best_state[j] == 1) sa_hospitals++;
            double sa_cost = calculate_total_cost(best_state);

            // Save a snapshot of SA result at λ = 50 for the map plot
            if (lambda == 50.0)
                memcpy(snapshot_sa, best_state, sizeof(int) * M);

            fprintf(f_metrics, "SA,%f,%d,%d,%f,%f\n",
                    lambda, run, sa_hospitals, sa_cost, time2);

            // ── Accumulate totals ─────────────────────────────────────────
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

        // ── Print averaged summary for this λ ────────────────────────────
        printf("\n--- Results for Lambda (λ) = %.1f (Averaged over 10 runs) ---\n", lambda);
        printf("  [Hill Climbing]\n");
        printf("    Avg Open Hospitals : %.1f\n",      hc_sum_hospitals / 10.0);
        printf("    Avg Objective Cost : %.2f\n",      hc_sum_cost      / 10.0);
        printf("    Avg Execution Time : %.4f seconds\n", hc_sum_time   / 10.0);

        printf("\n  [Simulated Annealing]\n");
        printf("    Avg Open Hospitals : %.1f\n",      sa_sum_hospitals / 10.0);
        printf("    Avg Objective Cost : %.2f\n",      sa_sum_cost      / 10.0);
        printf("    Avg Execution Time : %.4f seconds\n", sa_sum_time   / 10.0);

        printf("\n  [Head-to-Head Win Rate]\n");
        printf("    SA Wins: %d | HC Wins: %d | Ties: %d\n", sa_wins, hc_wins, ties);
        printf("------------------------------------------------------------------\n");
    }

    fclose(f_metrics);

    // ── Export candidate site selections (snapshot at λ = 50) ────────────────
    // Using λ = 50 gives a balanced, informative geographic view —
    // neither too many hospitals (low λ) nor too few (high λ).
    FILE *f_cand = fopen("candidates.csv", "w");
    if (!f_cand) { perror("Error creating candidates.csv"); return 1; }
    fprintf(f_cand, "X,Y,HC_Open,SA_Open\n");
    for (int j = 0; j < M; j++) {
        fprintf(f_cand, "%f,%f,%d,%d\n",
                candidates[j].x, candidates[j].y,
                snapshot_hc[j], snapshot_sa[j]);
    }
    fclose(f_cand);

    printf("\n Data successfully exported to CSV files for Python visualization!\n");
    printf(" (candidates.csv reflects solutions at lambda = 50)\n\n");

    return 0;
}