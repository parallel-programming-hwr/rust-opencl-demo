__kernel void check_prime(__global const long *IN, __global bool *OUT) {
    int id = get_global_id(0);
    int num = IN[id];
    bool prime = true;

    if (num < 3 || num % 2 == 0) {
        prime = false;
    } else {
        for (int i = 3; i <= sqrt((float) num); i += 2) {
            if (num % i == 0) {
                prime = false;
                break;
            }
        }
    }

    OUT[id] = prime;
}