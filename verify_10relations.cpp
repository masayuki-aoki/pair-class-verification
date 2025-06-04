%%writefile verify_relations.cpp
#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <omp.h>

// Constants
const long long MAX_TWO_N = 10000000;  // 10^7
const long long SEGMENT_SIZE = 1000000;

// Fast prime sieve implementation
std::vector<bool> build_prime_sieve(long long n_max) {
    std::vector<bool> is_prime(n_max + 1, true);
    is_prime[0] = is_prime[1] = false;
    
    // Mark even numbers (except 2) as composite
    for (long long i = 4; i <= n_max; i += 2) {
        is_prime[i] = false;
    }
    
    // Mark multiples of odd primes
    long long sqrt_n = std::sqrt(n_max) + 1;
    #pragma omp parallel for schedule(dynamic)
    for (long long i = 3; i <= sqrt_n; i += 2) {
        if (is_prime[i]) {
            for (long long j = i * i; j <= n_max; j += 2 * i) {
                is_prime[j] = false;
            }
        }
    }
    
    return is_prime;
}

// Prime counting function π(x)
long long pi_function(long long x, const std::vector<bool>& is_prime) {
    if (x < 2) return 0;
    
    long long count = 0;
    long long limit = std::min(x, static_cast<long long>(is_prime.size() - 1));
    
    for (long long i = 2; i <= limit; ++i) {
        if (is_prime[i]) {
            count++;
        }
    }
    
    return count;
}

// Twin prime counting function π₂(x)
long long pi2_function(long long x, const std::vector<bool>& is_prime) {
    if (x < 5) return 0;
    
    long long count = 0;
    long long limit = std::min(x - 2, static_cast<long long>(is_prime.size() - 3));
    
    for (long long p = 3; p <= limit; p += 2) {
        if (is_prime[p] && p + 2 < is_prime.size() && is_prime[p + 2]) {
            count++;
        }
    }
    
    return count;
}

// Compute class sizes
std::vector<long long> compute_class_sizes(long long two_n, const std::vector<bool>& is_prime) {
    std::vector<long long> counts(8, 0);
    
    for (long long p = 3; p < two_n - 2; p += 2) {
        long long q = two_n - p;
        if (q < 3 || q % 2 == 0) continue;
        
        bool p_is_prime = p < is_prime.size() && is_prime[p];
        bool q_is_prime = q < is_prime.size() && is_prime[q];
        bool qp2_is_prime = (q + 2) < is_prime.size() && is_prime[q + 2];
        
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

// Verify relations for a single 2n value
std::vector<int> verify_relations(long long two_n, 
                                 const std::vector<long long>& current_classes,
                                 const std::vector<long long>& prev_classes, 
                                 const std::vector<bool>& is_prime) {
    std::vector<int> failures;
    
    // Extract class sizes
    long long PPp = current_classes[0];
    long long PPq = current_classes[1];
    long long PQp = current_classes[2];
    long long PQq = current_classes[3];
    long long QPp = current_classes[4]; 
    long long QPq = current_classes[5];
    long long QQp = current_classes[6];
    long long QQq = current_classes[7];
    
    // Previous classes
    long long prev_PPp = prev_classes[0];
    long long prev_PPq = prev_classes[1];
    long long prev_PQp = prev_classes[2];
    long long prev_PQq = prev_classes[3];
    long long prev_QPp = prev_classes[4];
    long long prev_QPq = prev_classes[5];
    long long prev_QQp = prev_classes[6];
    long long prev_QQq = prev_classes[7];
    
    // Prime counting values
    long long pi = pi_function(two_n, is_prime);
    long long pi_minus_2 = pi_function(two_n - 2, is_prime);
    long long pi_minus_4 = pi_function(two_n - 4, is_prime);
    long long pi2_val = pi2_function(two_n, is_prime);
    long long n = two_n / 2;
    
    // Check R1: PPp + QPp = π₂(2n)
    if (!(PPp + QPp == pi2_val)) {
        failures.push_back(1);
    }
    
    // Check R2: PPq + QPq = π(2n-2) - π₂(2n) - 1
    if (!(PPq + QPq == pi_minus_2 - pi2_val - 1)) {
        failures.push_back(2);
    }
    
    // Check R3: PQp + QQp = π(2n) - π₂(2n) - 2
    if (!(PQp + QQp == pi - pi2_val - 2)) {
        failures.push_back(3);
    }
    
    // Check R4: PQq + QQq = n - π(2n) - π(2n-2) + π₂(2n) + 1
    if (!(PQq + QQq == n - pi - pi_minus_2 + pi2_val + 1)) {
        failures.push_back(4);
    }
    
    // Check R5: PPp + PPq + PQp + PQq = π(2n-2) - 1
    if (!(PPp + PPq + PQp + PQq == pi_minus_2 - 1)) {
        failures.push_back(5);
    }
    
    // Check R6: PQp + PQq = QPp + QPq (FIXED: current level, not previous!)
    if (!(PQp + PQq == QPp + QPq)) {
        failures.push_back(6);
    }
    
    // Check R7: PPp + PPq = PPp(2n-2) + PQp(2n-2) + [π(2n-2) - π(2n-4)]
    if (!(PPp + PPq == prev_PPp + prev_PQp + (pi_minus_2 - pi_minus_4))) {
        failures.push_back(7);
    }
    
    // Check R8: PQp + PQq = PPq(2n-2) + PQq(2n-2)
    if (!(PQp + PQq == prev_PPq + prev_PQq)) {
        failures.push_back(8);
    }
    
    // Check R9: QPp + QPq = QQp(2n-2) + QPp(2n-2) - [π(2n-2) - π(2n-4)] + 1
    if (!(QPp + QPq == prev_QQp + prev_QPp - (pi_minus_2 - pi_minus_4) + 1)) {
        failures.push_back(9);
    }
    
    // Check R10: QQp + QQq = QQq(2n-2) + QPq(2n-2)
    if (!(QQp + QQq == prev_QQq + prev_QPq)) {
        failures.push_back(10);
    }
    
    return failures;
}

// Main entry point
int main(int argc, char* argv[]) {
    long long max_two_n = MAX_TWO_N;
    if (argc > 1) {
        max_two_n = std::stoll(argv[1]);
    }
    
    std::cout << "Starting verification for 2n up to " << max_two_n << std::endl;
    std::cout << "Verifying 10 relations from Aoki's pair-class framework" << std::endl;
    
    // Start timer
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Build prime sieve
    long long sieve_size = max_two_n + 1000;
    std::cout << "Building prime sieve up to " << sieve_size << std::endl;
    
    std::vector<bool> is_prime = build_prime_sieve(sieve_size);
    
    auto sieve_end = std::chrono::high_resolution_clock::now();
    auto sieve_duration = std::chrono::duration_cast<std::chrono::seconds>(sieve_end - start_time).count();
    std::cout << "Prime sieve built in " << sieve_duration << "s" << std::endl;
    
    // Process in segments
    std::vector<std::pair<long long, int>> all_failures;
    
    // Initial computation for 2n=4
    std::vector<long long> prev_classes = compute_class_sizes(4, is_prime);
    
    // Process each segment
    for (long long start_two_n = 6; start_two_n <= max_two_n; start_two_n += SEGMENT_SIZE) {
        long long end_two_n = std::min(start_two_n + SEGMENT_SIZE, max_two_n + 2);
        
        std::cout << "Processing segment " << start_two_n << " to " << (end_two_n - 2) << "..." << std::endl;
        auto segment_start = std::chrono::high_resolution_clock::now();
        
        std::vector<std::pair<long long, int>> segment_failures;
        
        #pragma omp parallel for ordered schedule(dynamic) shared(is_prime, segment_failures)
        for (long long two_n = start_two_n; two_n < end_two_n; two_n += 2) {
            // Compute classes for current 2n
            std::vector<long long> current_classes = compute_class_sizes(two_n, is_prime);
            
            // Compute classes for previous 2n-2 if needed
            std::vector<long long> prev_classes_local;
            if (two_n == 6) {
                // Use precomputed classes for 2n=4
                prev_classes_local = prev_classes;
            } else {
                prev_classes_local = compute_class_sizes(two_n - 2, is_prime);
            }
            
            // Verify relations
            std::vector<int> failures = verify_relations(two_n, current_classes, prev_classes_local, is_prime);
            
            // Record failures (thread-safe)
            if (!failures.empty()) {
                #pragma omp ordered
                {
                    for (int relation : failures) {
                        segment_failures.push_back(std::make_pair(two_n, relation));
                    }
                }
            }
        }
        
        // Add segment failures to overall list
        for (const auto& failure : segment_failures) {
            all_failures.push_back(failure);
        }
        
        auto segment_end = std::chrono::high_resolution_clock::now();
        auto segment_duration = std::chrono::duration_cast<std::chrono::seconds>(segment_end - segment_start).count();
        std::cout << "Segment completed in " << segment_duration << "s" << std::endl;
        
        // Show progress
        double progress = 100.0 * (end_two_n - 6) / (max_two_n - 4);
        std::cout << "Progress: " << progress << "% complete" << std::endl;
        
        // Report segment failures
        if (!segment_failures.empty()) {
            std::cout << "Found " << segment_failures.size() << " failures in this segment" << std::endl;
            int limit = std::min(5, static_cast<int>(segment_failures.size()));
            for (int i = 0; i < limit; ++i) {
                std::cout << "  2n=" << segment_failures[i].first 
                          << ", relation R" << segment_failures[i].second << " failed" << std::endl;
            }
            if (segment_failures.size() > 5) {
                std::cout << "  ... and " << (segment_failures.size() - 5) 
                          << " more failures in this segment" << std::endl;
            }
        }
    }
    
    // Measure total time
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    
    // Report results
    std::cout << "Verification completed in " << total_duration << "s" << std::endl;
    int hours = total_duration / 3600;
    int minutes = (total_duration % 3600) / 60;
    int seconds = total_duration % 60;
    std::cout << "Total time: " << hours << "h " << minutes << "m " << seconds << "s" << std::endl;
    
    if (all_failures.empty()) {
        std::cout << "All ten relations verified ✓ for 2n up to " << max_two_n << std::endl;
    } else {
        // Group failures by 2n
        std::unordered_map<long long, std::vector<int>> failure_map;
        
        for (const auto& failure : all_failures) {
            failure_map[failure.first].push_back(failure.second);
        }
        
        std::cout << "Found failures for " << failure_map.size() << " different 2n values:" << std::endl;
        
        // Convert to vector for sorting
        std::vector<std::pair<long long, std::vector<int>>> sorted_failures;
        for (const auto& entry : failure_map) {
            sorted_failures.push_back({entry.first, entry.second});
        }
        
        // Sort by 2n value
        std::sort(sorted_failures.begin(), sorted_failures.end(),
                 [](const auto& a, const auto& b) { return a.first < b.first; });
        
        // Print first 10 failures
        int limit = std::min(10, static_cast<int>(sorted_failures.size()));
        for (int i = 0; i < limit; ++i) {
            const auto& entry = sorted_failures[i];
            long long two_n = entry.first;
            std::vector<int> relations = entry.second;
            
            // Sort relation numbers
            std::sort(relations.begin(), relations.end());
            
            std::cout << "  2n=" << two_n << ": Failed relations [";
            for (size_t j = 0; j < relations.size(); ++j) {
                if (j > 0) std::cout << ", ";
                std::cout << "R" << relations[j];
            }
            std::cout << "]" << std::endl;
        }
        
        if (sorted_failures.size() > 10) {
            std::cout << "  ... and " << (sorted_failures.size() - 10) << " more failures." << std::endl;
        }
    }
    
    return 0;
}