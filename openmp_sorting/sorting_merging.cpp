#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <omp.h>
#include <algorithm>
#include <queue>
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

        // each thread when inserting elements from original array into buckets
        // has its own threads, so let's assume that each thread has 3 buckets
        // then the zeroed thread will have buckets with indices 0, 1, 2
        // the first thread 3, 4, 5
        // and generally thread_id will have:
        // number_of_buckets*thread_id, number_of_buckets*thread_id+1, ... number_of_buckets*thread_id+(number_of-buckets)-1
        int total_number_of_buckets = number_of_threads*number_of_buckets;

        vector<int>* buckets[total_number_of_buckets];

        priority_queue<int> *queues[number_of_buckets];


        for(int i=0; i<total_number_of_buckets; i++) {
                buckets[i] = new vector<int>();
        }

        for(int i=0; i<number_of_buckets; i++) {
            queues[i] = new priority_queue<int>() ;
        }

        // first part is the same as in normal sorting -> random inserting into original array
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

        // here comes the fun part!
        #pragma omp parallel for
        for(int i=0; i<number_of_points; i++) {
            int thread_id = omp_get_thread_num();
            // let's assume we have merged buckets for a moment, to calculate offset of bucket where to insert that particular element
            int divide = range_of_numbers / number_of_buckets;
                if (divide == 0) {
                        divide = range_of_numbers;
            }
            int bucket_offset = elementTable[i]/divide;
            bucket_offset = min(bucket_offset, number_of_buckets-1);
            int offset_group_start_number = thread_id * number_of_buckets;
            buckets[offset_group_start_number + bucket_offset]->push_back(elementTable[i]);
        }

        if(TIME_DEBUG) {
          counter_new = omp_get_wtime();
          cout << "splitting to buckets: " << counter_new-counter_old << endl;
          counter_old = omp_get_wtime();
        }

        #pragma omp parallel for
        for(int i=0; i<total_number_of_buckets; i++) {
            sort(buckets[i]->begin(), buckets[i]->end());
        }

        if(TIME_DEBUG) {
          counter_new = omp_get_wtime();
          cout << "sorting buckets: " << counter_new-counter_old << endl;
          counter_old = omp_get_wtime();
        }

        // merging!
        #pragma omp parallel for
        for(int i=0; i<number_of_buckets; i++) {
            for(int bucket=i; bucket<total_number_of_buckets; bucket+=number_of_buckets) {
                for(int element_idx=0; element_idx<buckets[bucket]->size(); element_idx+=1) {
                    queues[i]->push(-buckets[bucket]->at(element_idx));
                }
            }
        }

        vector<int> offset_vector;

        int current_sum = 0;
        for(int i=0; i<number_of_buckets; i++) {
            current_sum += queues[i]->size();
            offset_vector.push_back(current_sum);
        }

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


                for(int k=get_offset_start; k<get_offset_end; k++) {
                        result_vector->at(k) = queues[i]->top();
                        queues[i]->pop();
                }
        }

        if(TIME_DEBUG) {
          counter_new = omp_get_wtime();
          cout << "merging buckets: " << counter_new-counter_old << endl;
          counter_old = omp_get_wtime();
        }

        // for(int i=0; i<result_vector->size(); i++) {
        //     cout << -result_vector->at(i) << " ";
        // }


        double end = omp_get_wtime();
        cout << end-start << "," << original_number_of_points << "," << number_of_buckets << "," << range_of_numbers << "," << number_of_threads << endl;

        delete [] elementTable;

        for(int i=0; i<total_number_of_buckets; i++) {
            delete buckets[i];
        }

        for(int i=0; i<number_of_buckets; i++) {
            delete queues[i];
        }

        delete result_vector;

        return 0;
}
