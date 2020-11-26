/*
 * opencl demos with rust
 * Copyright (C) 2020 trivernis
 * See LICENSE for more information
 */


__kernel void check_prime(const int LOWER_PRIME_COUNT, __global const long *LOWER_PRIMES, __global const long *IN, __global bool *OUT) {
    int id = get_global_id(0);
    long num = IN[id];
    bool prime = true;
    long limit = (long) sqrt((double) num) + 1;

    if (num < 3 || num % 2 == 0) {
        prime = false;
    } else {
        for (int i = 0; i < LOWER_PRIME_COUNT; i++) {
            if (LOWER_PRIMES[i] >= limit) {
                break;
            }
            if (num % LOWER_PRIMES[i] == 0) {
                prime = false;
                break;
            }
        }
        long start = LOWER_PRIMES[LOWER_PRIME_COUNT - 1];
        start -= start % 3;

        if (start % 2 == 0) {
            start -= 3;
        }

        if (prime && start < limit) {
            for (long i = start; i <= limit; i += 6) {
                if (num % (i - 2) == 0 || num % (i - 4) == 0) {
                    prime = false;
                    break;
                }
            }
        }
    }

    OUT[id] = prime;
}