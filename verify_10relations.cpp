%%writefile verify_relations_checkpoint.cpp
// verify_relations_checkpoint.cpp
// Verification of the ten pair-class relations (pair-class framework)
// Designed for Google Colab: fast, memory-efficient, checkpointed
// Author: Masayuki Aoki (with ChatGPT refinements)
// Last Update: 2025-06

#include <iostream>
#include <vector>
#include <array>
#include <bitset>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <omp.h>
#include <exception>

const long long MAX_TWO_N = 10000000;     // Maximum value of 2n to check (adjustable)
const long long SEGMENT_SIZE = 100000;    // How many 2n per segment/checkpoint
const std::string CHECKPOINT_FILE = "verification_checkpoint.txt";
const std::string FAILURES_FILE    = "verification_failures.txt";

// Use bitset for is_prime (memory efficient, fast lookup)
// (bitset max size = 1e7+2000 is ~1.3MB)
const long long TABLE_SIZE = MAX_TWO_N + 2000;
std::bitset<TABLE_SIZE> is_prime;
std::vector<int> pi_arr;   // π(x) lookup
std::vector<int> pi2_arr;  // π₂(x) lookup

// Checkpoint structure for safe restart
struct Checkpoint {
    long long last_completed_segment = 4; // Last completed segment's starting 2n
    long long total_failures = 0;
    long long total_checked = 0;

    void save() const {
        std::ofstream file(CHECKPOINT_FILE);
        if (file.is_open()) {
            file << last_completed_segment << " " << total_failures << " " << total_checked << std::endl;
            file.close();
            std::cout << "[Checkpoint] Saved at segment " << last_completed_segment << std::endl;
        }
    }
    bool load() {
        std::ifstream file(CHECKPOINT_FILE);
        if (file.is_open()) {
            file >> last_completed_segment >> total_failures >> total_checked;
            file.close();
            return true;
        }
        return false;
    }
};

// Build prime table, π(x), and π₂(x)
void build_prime_tables(long long n_max) {
    is_prime.reset();
    is_prime[2] = true;
    for (long long i = 3; i <= n_max; i += 2) is_prime[i] = true;

    long long sqrt_n = std::sqrt(n_max) + 1;
    for (long long p = 3; p <= sqrt_n; p += 2) {
        if (is_prime[p]) {
            for (long long j = p * p; j <= n_max; j += 2 * p) is_prime[j] = false;
        }
    }

    pi_arr.assign(n_max + 1, 0);
    pi2_arr.assign(n_max + 1, 0);

    int cnt_pi = 0, cnt_pi2 = 0;
    for (long long x = 2; x <= n_max; ++x) {
        if (is_prime[x]) cnt_pi++;
        pi_arr[x] = cnt_pi;
    }
    for (long long x = 2; x <= n_max; ++x) {
        if (x >= 5 && is_prime[x - 2] && is_prime[x]) cnt_pi2++;
        pi2_arr[x] = cnt_pi2;
    }
}

// Fast lookups for π(x) and π₂(x)
inline int Pi(long long x) {
    if (x < 2 || x >= pi_arr.size()) return 0;
    return pi_arr[x];
}
inline int Pi2(long long x) {
    if (x < 2 || x >= pi2_arr.size()) return 0;
    return pi2_arr[x];
}

// Compute class counts for a given 2n
std::array<long long, 8> compute_class_sizes(long long two_n) {
    std::array<long long, 8> counts = {0, 0, 0, 0, 0, 0, 0, 0};
    // Loop over odd p
    for (long long p = 3; p < two_n - 2; p += 2) {
        long long q = two_n - p;
        if (q < 3 || (q % 2 == 0)) continue;
        bool p_is_prime = p < TABLE_SIZE && is_prime[p];
        bool q_is_prime = q < TABLE_SIZE && is_prime[q];
        bool qp2_is_prime = (q + 2) < TABLE_SIZE && is_prime[q + 2];
        int idx = 0;
        if (p_is_prime) {
            if (q_is_prime) {
                idx = qp2_is_prime ? 0 : 1;  // PPp or PPq
            } else {
                idx = qp2_is_prime ? 2 : 3;  // PQp or PQq
            }
        } else {
            if (q_is_prime) {
                idx = qp2_is_prime ? 4 : 5;  // QPp or QPq
            } else {
                idx = qp2_is_prime ? 6 : 7;  // QQp or QQq
            }
        }
        counts[idx]++;
    }
    return counts;
}

// Verify the ten pair-class relations for a single 2n value
std::vector<int> verify_relations(long long two_n,
                                 const std::array<long long, 8>& current,
                                 const std::array<long long, 8>& prev) {
    std::vector<int> failures;
    long long PPp = current[0], PPq = current[1];
    long long PQp = current[2], PQq = current[3];
    long long QPp = current[4], QPq = current[5];
    long long QQp = current[6], QQq = current[7];

    long long prev_PPp = prev[0], prev_PPq = prev[1];
    long long prev_PQp = prev[2], prev_PQq = prev[3];
    long long prev_QPp = prev[4], prev_QPq = prev[5];
    long long prev_QQp = prev[6], prev_QQq = prev[7];

    int pi = Pi(two_n);
    int pi_minus_2 = Pi(two_n - 2);
    int pi_minus_4 = Pi(two_n - 4);
    int pi2_val = Pi2(two_n);
    long long n = two_n / 2;

    if (PPp + QPp != pi2_val) failures.push_back(1);
    if (PPq + QPq != pi_minus_2 - pi2_val - 1) failures.push_back(2);
    if (PQp + QQp != pi - pi2_val - 2) failures.push_back(3);
    if (PQq + QQq != n - pi - pi_minus_2 + pi2_val + 1) failures.push_back(4);
    if (PPp + PPq + PQp + PQq != pi_minus_2 - 1) failures.push_back(5);
    if (PQp + PQq != QPp + QPq) failures.push_back(6);

    if (two_n > 6) {
        if (PPp + PPq != prev_PPp + prev_PQp + (pi_minus_2 - pi_minus_4)) failures.push_back(7);
        if (PQp + PQq != prev_PPq + prev_PQq) failures.push_back(8);
        if (QPp + QPq != prev_QQp + prev_QPp - (pi_minus_2 - pi_minus_4) + 1) failures.push_back(9);
        if (QQp + QQq != prev_QQq + prev_QPq) failures.push_back(10);
    }
    return failures;
}

int main(int argc, char* argv[]) {
    long long max_two_n = MAX_TWO_N;
    long long start_from = 6;  // Default start

    // Check command line arguments for start and end values
    if (argc > 1) {
        max_two_n = std::stoll(argv[1]);
    }
    if (argc > 2) {
        start_from = std::stoll(argv[2]);
        if (start_from < 6) start_from = 6;
        if (start_from % 2 == 1) start_from++;
    }

    // Load checkpoint if available (Colab/Jupyter: always use command-line resume!)
    Checkpoint checkpoint;
    bool resumed = false;
    if (checkpoint.load()) {
        std::cout << "Checkpoint found at segment " << checkpoint.last_completed_segment << ".\n";
        std::cout << "Previous: " << checkpoint.total_checked << " checked, "
                  << checkpoint.total_failures << " failures.\n";
        std::cout << "[Colab/Jupyter mode] To resume, use:\n";
        std::cout << "  ./verify_checkpoint " << max_two_n << " " << (checkpoint.last_completed_segment + SEGMENT_SIZE) << std::endl;
        std::cout << "Starting from 6 (fresh) by default.\n";
        // Only auto-resume if user specified a 2nd argument (see above).
        if (argc <= 2) {
            start_from = 6;
            resumed = false;
        } else if (start_from > checkpoint.last_completed_segment) {
            resumed = true;
            std::cout << "Resuming from " << start_from << std::endl;
        }
    }

    std::cout << "Verifying 2n from " << start_from << " to " << max_two_n << std::endl;
    std::cout << "Segment size: " << SEGMENT_SIZE << std::endl;
    std::cout << "OpenMP threads: " << omp_get_max_threads() << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();
    std::cout << "Building prime tables up to " << TABLE_SIZE << "..." << std::endl;
    build_prime_tables(TABLE_SIZE);
    auto build_end = std::chrono::high_resolution_clock::now();
    auto build_dur = std::chrono::duration_cast<std::chrono::seconds>(build_end - start_time).count();
    std::cout << "Tables built in " << build_dur << "s\n";

    std::ofstream failures_file(FAILURES_FILE, resumed ? std::ios::app : std::ios::out);
    if (!resumed) {
        failures_file << "# 2n, relation_number" << std::endl;
    }

    long long total_failures = resumed ? checkpoint.total_failures : 0;
    long long total_checked  = resumed ? checkpoint.total_checked : 0;

    try {
        for (long long segment_start = start_from; segment_start <= max_two_n; segment_start += SEGMENT_SIZE) {
            long long segment_end = std::min(segment_start + SEGMENT_SIZE, max_two_n + 2);

            std::cout << "\n[Segment] " << segment_start << " to " << (segment_end - 2) << "..." << std::endl;
            auto seg_start_time = std::chrono::high_resolution_clock::now();

            std::vector<std::pair<long long, int>> segment_failures;
            long long segment_checked = 0;

            #pragma omp parallel for schedule(dynamic, 1000) reduction(+:segment_checked)
            for (long long two_n = segment_start; two_n < segment_end; two_n += 2) {
                auto current = compute_class_sizes(two_n);
                auto prev    = compute_class_sizes(two_n - 2);

                auto failures = verify_relations(two_n, current, prev);

                if (!failures.empty()) {
                    #pragma omp critical
                    {
                        for (int rel : failures) {
                            segment_failures.emplace_back(two_n, rel);
                            failures_file << two_n << "," << rel << std::endl;
                        }
                    }
                }
                segment_checked++;
            }

            total_failures += segment_failures.size();
            total_checked  += segment_checked;

            auto seg_end_time = std::chrono::high_resolution_clock::now();
            auto seg_dur = std::chrono::duration_cast<std::chrono::milliseconds>(seg_end_time - seg_start_time).count();

            double progress = 100.0 * (segment_end - 6) / (max_two_n - 4);
            std::cout << "Segment done in " << seg_dur/1000.0 << "s. Progress: "
                      << std::fixed << std::setprecision(1) << progress << "%, Speed: "
                      << segment_checked / (seg_dur/1000.0) << " vals/sec" << std::endl;
            if (!segment_failures.empty())
                std::cout << "  [!] Failures in segment: " << segment_failures.size() << std::endl;

            std::cout << "  Total: " << total_checked << " checked, " << total_failures << " failures\n";

            // Checkpoint and flush
            checkpoint.last_completed_segment = segment_start;
            checkpoint.total_failures = total_failures;
            checkpoint.total_checked  = total_checked;
            checkpoint.save();
            failures_file.flush();
        }
    } catch (const std::exception& e) {
        std::cerr << "[Error] Exception caught: " << e.what() << std::endl;
        checkpoint.save();
        failures_file.flush();
        return 2;
    } catch (...) {
        std::cerr << "[Error] Unknown exception caught." << std::endl;
        checkpoint.save();
        failures_file.flush();
        return 3;
    }

    failures_file.close();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_dur = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();

    std::cout << "\n=== VERIFICATION COMPLETE ===\n";
    std::cout << "Checked: " << total_checked << ", Time: " << total_dur << "s\n";
    std::cout << "Total failures: " << total_failures << std::endl;

    if (total_failures == 0) {
        std::cout << "\n✓ SUCCESS: All relations verified!" << std::endl;
        std::remove(CHECKPOINT_FILE.c_str());
    } else {
        std::cout << "\n✗ Failures found. See " << FAILURES_FILE << " for details." << std::endl;
    }
    return total_failures == 0 ? 0 : 1;
}
// !g++ -std=c++17 -O3 -fopenmp verify_relations_checkpoint.cpp -o verify_checkpoint
// !./verify_checkpoint 10000000
