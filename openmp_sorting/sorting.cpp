#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <omp.h>
#include <algorithm>
#include <random>

#define DEBUG true
#define TIME_DEBUG true

using namespace std;

int main(int argc, char**argv) {
        if (argc != 6) {
                cout << "Please provide arguments [number_of_points number_of_buckets range_of_numbers number_of_threads is_scalable]" << endl;
                return -1;
        }

        double start = omp_get_wtime();

        int number_of_points = atoi(argv[1]);
        int number_of_buckets = atoi(argv[2]);
        int range_of_numbers = atoi(argv[3]);
        int number_of_threads = atoi(argv[4]);

        int original_number_of_points = number_of_points;

        if(string(argv[5])=="scalable") {
          number_of_points = number_of_points * number_of_threads;
        }

        omp_set_num_threads(number_of_threads);

        double counter_new;
        double counter_old;

        if(TIME_DEBUG) counter_old = omp_get_wtime();

        int *elementTable = new int[number_of_points];

        vector<int>* buckets[number_of_buckets];
        for(int i=0; i<number_of_buckets; i++) {
                buckets[i] = new vector<int>();
        }
        omp_lock_t write_locks[number_of_buckets];

        for(int i=0; i<number_of_buckets; i++) {
                omp_init_lock(&write_locks[i]);
        }

        #pragma omp parallel
        {
          std::random_device rd;
          std::mt19937 gen(rd());
          std::uniform_int_distribution<> dis(0, range_of_numbers);

          #pragma omp for
          for(int n=0; n<number_of_points; ++n) {
            elementTable[n] = dis(gen);
          }
        }

        if(TIME_DEBUG) {
          counter_new = omp_get_wtime();
          cout << "table generation: " << counter_new-counter_old << endl;
          counter_old = omp_get_wtime();
        }

 //       if (DEBUG) {
 //               for(int i=0; i<number_of_points; i++) {
 //                       cout << elementTable[i] << " ";
 //               }
 //       }

        #pragma omp parallel for
        for(int n=0; n<number_of_points; ++n) {
                int divide = range_of_numbers / number_of_buckets;
                if (divide == 0) {
                        divide = range_of_numbers;
                }
                int bucket_number = elementTable[n]/divide;
                bucket_number = min(bucket_number, number_of_buckets-1);
                omp_set_lock(&write_locks[bucket_number]);
                buckets[bucket_number]->push_back(elementTable[n]);
                omp_unset_lock(&write_locks[bucket_number]);
        }

        for(int i=0; i<number_of_buckets; i++) {
                omp_destroy_lock(&write_locks[i]);
        }

        if(TIME_DEBUG) {
          counter_new = omp_get_wtime();
          cout << "splitting to buckets: " << counter_new-counter_old << endl;
          counter_old = omp_get_wtime();
        }

        // if (DEBUG) {
        //         for(int i=0; i<number_of_buckets; i++) {
        //                 for(int j=0; j<buckets[i].size(); j++) {
        //                         cout << buckets[i][j] << " ";
        //                 }
        //                 cout << endl;
        //         }
        // }

        #pragma omp parallel for
        for(int i=0; i<number_of_buckets; i++) {

                sort(buckets[i]->begin(), buckets[i]->end());
        }

        if(TIME_DEBUG) {
          counter_new = omp_get_wtime();
          cout << "sorting buckets: " << counter_new-counter_old << endl;
          counter_old = omp_get_wtime();
        }

        vector<int> offset_vector;
        int current_sum = 0;

        for(int i=0; i<number_of_buckets; i++) {
                int bucket_size = buckets[i]->size();
                current_sum += bucket_size;
                offset_vector.push_back(current_sum);
        }
        // if (DEBUG) {
        //         for(int j=0; j<offset_vector.size(); j++) {
        //                 cout << offset_vector[j] << " ";
        //         }
        // }

        vector<int>* result_vector = new vector<int>(current_sum);

        #pragma omp parallel for
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
                        result_vector->at(k) = buckets[i]->at(element_in_bucket_idx);
                        element_in_bucket_idx+=1;
                }
        }

        if(TIME_DEBUG) {
          counter_new = omp_get_wtime();
          cout << "merging buckets: " << counter_new-counter_old << endl;
          counter_old = omp_get_wtime();
        }

        // if (DEBUG) {
        //         for(auto x : result_vector) {
        //                 cout << x << " ";
        //         }
        // }

        double end = omp_get_wtime();
        cout << end-start << "," << original_number_of_points << "," << number_of_buckets << "," << range_of_numbers << "," << number_of_threads << endl;
       	delete[] elementTable;
//	delete[] buckets;
	delete result_vector;
        return 0;

}
