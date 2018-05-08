#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <omp.h>
#include <algorithm>
#define DEBUG false

using namespace std;

int main(int argc, char**argv) {
        if (argc != 5) {
                cout << "Please provide arguments [number_of_points number_of_buckets range_of_numbers number_of_threads]" << endl;
                return -1;
        }

        int number_of_points = stoi(argv[1]);
        int number_of_buckets = stoi(argv[2]);
        int range_of_numbers = stoi(argv[3]);
        int number_of_threads = stoi(argv[4]);

        int elementTable[number_of_points];
        
        vector<int> buckets[number_of_buckets];
        omp_lock_t write_locks[number_of_buckets];

        for(int i=0; i<number_of_buckets; i++) {
                omp_init_lock(&write_locks[i]);
        }

        #pragma omp parallel  num_threads(number_of_threads)
        {
                srand(int(time(NULL)) ^ omp_get_thread_num()); // https://www.viva64.com/en/b/0012/
                #pragma omp for
                for(int n=0; n<number_of_points; ++n) {
                        int rand_number = rand() % range_of_numbers;
                        elementTable[n] = rand_number;
                }
        }

        if (DEBUG) {
                for(int i=0; i<number_of_points; i++) {
                        cout << elementTable[i] << " ";
                }
        }

        #pragma omp parallel for num_threads(number_of_threads)
        for(int n=0; n<number_of_points; ++n) {
                int divide = range_of_numbers / number_of_buckets;
                if (divide == 0) {
                        divide = range_of_numbers;
                }
                int bucket_number = elementTable[n]/divide;
                bucket_number = min(bucket_number, number_of_buckets-1);
                omp_set_lock(&write_locks[bucket_number]);
                buckets[bucket_number].push_back(elementTable[n]);
                omp_unset_lock(&write_locks[bucket_number]);
        }

        for(int i=0; i<number_of_buckets; i++) {
                omp_destroy_lock(&write_locks[i]);
        }

        if (DEBUG) {
                for(int i=0; i<number_of_buckets; i++) {
                        for(int j=0; j<buckets[i].size(); j++) {
                                cout << buckets[i][j] << " ";
                        }
                        cout << endl;
                }
        }

        #pragma omp parallel for num_threads(number_of_threads)
        for(int i=0; i<number_of_buckets; i++) {
                
                sort(buckets[i].begin(), buckets[i].end());
        }

        vector<int> offset_vector;
        int current_sum = 0;

        for(int i=0; i<number_of_buckets; i++) {
                int bucket_size = buckets[i].size();
                current_sum += bucket_size;
                offset_vector.push_back(current_sum);
        }

        if (DEBUG) {
                for(int j=0; j<offset_vector.size(); j++) {
                        cout << offset_vector[j] << " ";
                }
        }

        vector<int> result_vector(current_sum);

        #pragma omp parallel for num_threads(number_of_threads)
        for(int i=0; i<number_of_buckets; i++) {
                int get_offset_start;
                if (i==0) {
                        get_offset_start = 0;
                } else {
                get_offset_start = offset_vector[i-1];
                }       
                int get_offset_end = offset_vector[i];

                int element_in_bucket_idx = 0;

                for(int k=get_offset_start; k<get_offset_end; k++) {
                        result_vector[k] = buckets[i][element_in_bucket_idx];
                        element_in_bucket_idx+=1;
                }
        }

        if (DEBUG) {
                for(auto x : result_vector) {
                        cout << x << " ";
                }
        }
        return 0;

}