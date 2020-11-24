__kernel void check_prime(const int LOWER_PRIME_COUNT, __global const int *LOWER_PRIMES, __global const long *IN, __global bool *OUT) {
    int id = get_global_id(0);
    long num = IN[id];
    bool prime = true;

    if (num < 3 || num % 2 == 0) {
        prime = false;
    } else {
        for (int i = 0; i < LOWER_PRIME_COUNT; i++) {
            if (LOWER_PRIMES[i] >= num) {
                break;
            }
            if (num % LOWER_PRIMES[i] == 0) {
                prime = false;
                break;
            }
        }
        if (prime && LOWER_PRIMES[LOWER_PRIME_COUNT - 1] < num) {
            for (long i = LOWER_PRIMES[LOWER_PRIME_COUNT - 1]; i <= sqrt((double) num); i += 2) {
                if (num % i == 0) {
                    prime = false;
                    break;
                }
            }
        }
    }

    OUT[id] = prime;
}