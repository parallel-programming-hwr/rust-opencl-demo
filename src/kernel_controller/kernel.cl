/*
 * opencl demos with rust
 * Copyright (C) 2020 trivernis
 * See LICENSE for more information
 */


__kernel void check_prime_cached(const uint LOWER_PRIME_COUNT, __global const ulong *LOWER_PRIMES, __global const ulong *IN, __global bool *OUT) {
    uint id = get_global_id(0);
    ulong num = IN[id];
    ulong limit = (ulong) native_sqrt((double) num) + 1;

    if (num < 3 || num % 2 == 0) {
        return;
    } else {
        for (uint i = 0; i < LOWER_PRIME_COUNT; i++) {
            if (LOWER_PRIMES[i] >= limit) {
                break;
            }
            if (num % LOWER_PRIMES[i] == 0) {
                return;
            }
        }
    }

    OUT[id] = true;
}

__kernel void check_prime(__global const ulong *IN, __global bool *OUT) {
    uint id = get_global_id(0);
    ulong num = IN[id];
    ulong limit = (ulong) native_sqrt((double) num) + 1;

    if (num < 3 || num % 2 == 0) {
        return;
    } else {
        for (ulong i = 9; i <= limit; i += 6) {
            if (num % (i - 2) == 0 || num % (i - 4) == 0) {
                return;
            }
        }
    }

    OUT[id] = true;
}