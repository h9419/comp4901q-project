#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <algorithm>
#include <random>

void sequentialPropagateWFC(const int &map_size, const int &x, const int &y,
                  std::vector<std::vector<int8_t>> *map,
                  std::vector<std::vector<int8_t>> *lower_bound,
                  std::vector<std::vector<int8_t>> *upper_bound) {
  int collapsed_value = map[0][x][y];
  // go through all cells
  for (int i=-9; i <= 9; ++i){
    for (int j=-9; j <= 9; ++j){
      // filter by manhattan distance
      int manhattan_distance = abs(i) + abs(j);
      if (manhattan_distance <= 9) {
        int xpos = x + i;
        int ypos = y + j;
        // map boundary
        if (xpos >= 0 && xpos < map_size && ypos >= 0 && ypos < map_size) {
          // non-determinate
          if (map[0][xpos][ypos] == -1) {
            lower_bound[0][xpos][ypos] = std::max((int)lower_bound[0][xpos][ypos], collapsed_value - manhattan_distance);
            upper_bound[0][xpos][ypos] = std::min((int)upper_bound[0][xpos][ypos], collapsed_value + manhattan_distance);
            // only 1 state left, set state
            if (lower_bound[0][xpos][ypos] == upper_bound[0][xpos][ypos]) {
              map[0][xpos][ypos] = upper_bound[0][xpos][ypos];
            }
          }
        }
      }
    }
  }
}

void sequentialWFC(const int &map_size, std::vector<std::vector<int8_t>> *map) {
  std::vector<std::vector<int8_t>> lower_bound, upper_bound;
  lower_bound = std::vector<std::vector<int8_t>>(map_size, std::vector<int8_t>(map_size,0));
  upper_bound = std::vector<std::vector<int8_t>>(map_size, std::vector<int8_t>(map_size,9));
  (*map) = std::vector<std::vector<int8_t>>(map_size, std::vector<int8_t>(map_size,-1));
  srand(time(NULL));

  // big primes
  // https://stackoverflow.com/a/18994414
  const unsigned int BIG_PRIME_X = 74207281;
  const unsigned int BIG_PRIME_Y = 74207279;
  unsigned int x = rand();
  unsigned int y = rand();
  // go through the map to set all cells
  for (int i=0; i<map_size; ++i) {
    y = (y + BIG_PRIME_Y) % map_size;
    for (int j=0; j<map_size; ++j) {
      x = (x + BIG_PRIME_X) % map_size;
      y = (y + BIG_PRIME_Y) % map_size;
      if ((*map)[x][y] == -1) {
        // set a cell randomly;
        (*map)[x][y] = lower_bound[x][y] + abs(rand() % (1 + upper_bound[x][y] - lower_bound[x][y]));
        // propagate changes
        sequentialPropagateWFC(map_size, x, y, map, &lower_bound, &upper_bound);
      }
    }
  }
}

// void SequentialCalculation(const int &n,
//                            const int &m,
//                            const std::vector<std::vector<int>> &A,
//                            const std::vector<std::vector<int>> &B,
//                            std::vector<std::vector<int>> *C) {

//   std::vector<std::vector<int>> B_power, next_B_power;
//   std::vector<std::vector<int>> D;
//   (*C) = A;
//   B_power = B;
//   int tmp;
//   for (int t = 1; t<=m; t++) {
//     D = std::vector<std::vector<int>>(n, std::vector<int>(n,0));
//     for (int i = 0; i<n; i++) {
//       for (int j = 0; j<n; j++) {
//         for (int k = 0; k<n; k++) {
//           D[i][j] = (D[i][j] + A[i][k] * B_power[k][j])%2;
//         }
//       } 
//     }
//     for (int i = 0; i<n; i++) {
//       for (int j = 0; j<n; j++) {
//         (*C)[i][j] = ((*C)[i][j] + D[i][j]) %2; 
//       }
//     } 
//     if (t==m)
//       break;
//     next_B_power = std::vector<std::vector<int>>(n, std::vector<int>(n,0));
//     for (int i = 0; i<n; i++) {
//       for (int j = 0; j<n; j++) {
//         for (int k = 0; k<n; k++)
//           next_B_power[i][j] = (next_B_power[i][j]+ B_power[i][k]*B[k][j])%2;
//       } 
//     }
//     B_power = next_B_power;
//   }
// }

bool TestAnswerCorrectness(const int &map_size, const std::vector<std::vector<int8_t>> &answer) {
  if (answer.size() != map_size) {
    std::cout << "Error! The answer size is incorrect" << std::endl;
    return false;
  }
  for (uint i = 0; i < map_size; i++) {
    if (answer[i].size() != map_size) {
      std::cout << "Error! The answer size is incorrect" << std::endl;
      return false;
    }
  }
  for (uint i = 0; i < map_size; i++) {
    for (uint j = 0; j < map_size; j++) {
      if (i > 0 && abs(answer[i][j] - answer[i-1][j]) > 1)
          return false;
      if (i < 9 && abs(answer[i][j] - answer[i+1][j]) > 1)
          return false;
      if (j > 0 && abs(answer[i][j] - answer[i][j-1]) > 1)
          return false;
      if (j < 9 && abs(answer[i][j] - answer[i][j+1]) > 1)
          return false;
    }
  }
  return true;
}

// This uses raw encoding with no compression or look up table because I simply needed something
// that works quickly and is compatible with standard image formats for it to be imported
void SaveImage(const int &map_size, const std::vector<std::vector<int8_t>> &answer) {
  const int BYTES_PER_PIXEL = 3; // RGB
  const int FILE_HEADER_SIZE = 14;
  const int INFO_HEADER_SIZE = 40;
  // Create a new file for writing
  std::ofstream fd("saved_result.bmp", std::ios_base::binary | std::ios_base::trunc);
  if (!fd.is_open()) {
      return;
  }
  int widthInBytes = map_size * BYTES_PER_PIXEL;
  int paddingSize = (4 - (widthInBytes) % 4) % 4;
  int stride = widthInBytes + paddingSize;

  int fileSize = FILE_HEADER_SIZE + INFO_HEADER_SIZE + (stride * map_size);
  const char fileHeader[FILE_HEADER_SIZE] = {
      // signature
      'B','M',
      // image file size in bytes
      (unsigned char)(fileSize),
      (unsigned char)(fileSize >> 8),
      (unsigned char)(fileSize >> 16),
      (unsigned char)(fileSize >> 24),
      // reserved
      0,0,0,0,
      // start of pixel array
      (unsigned char)(FILE_HEADER_SIZE + INFO_HEADER_SIZE),
      0,0,0
  };
  fd.write(fileHeader, FILE_HEADER_SIZE);
  const char infoHeader[INFO_HEADER_SIZE] = {
    // header size
    (unsigned char)(INFO_HEADER_SIZE), 0, 0, 0,
    // image width
    (unsigned char)(map_size),
    (unsigned char)(map_size >>  8),
    (unsigned char)(map_size >> 16),
    (unsigned char)(map_size >> 24),
    // image height
    (unsigned char)(map_size),
    (unsigned char)(map_size >>  8),
    (unsigned char)(map_size >> 16),
    (unsigned char)(map_size >> 24),
    // number of color planes
    1, 0,
    // bits per pixel
    (unsigned char)(BYTES_PER_PIXEL << 3), 0,
        0,0,0,0, // compression
        0,0,0,0, // image size
        0,0,0,0, // horizontal resolution
        0,0,0,0, // vertical resolution
        0,0,0,0, // colors in color table
        0,0,0,0, // important color count
  };
  fd.write(infoHeader, INFO_HEADER_SIZE);

  for (int i = 0; i < map_size; i++) {
    for (int j = 0; j < map_size; j++) {
      const char val = answer[i][j] * 28;
      for (int k = 0; k < 3; k++) {
        fd.write(&val, 1);
      }
    }
    const char padding[] = {0, 0, 0};
    fd.write(padding, paddingSize);
  }
  fd.close();
}

// // Device code
// __global__ void zero_matrix(int *matrix, int n) {
//   int index = blockDim.x * blockIdx.x + threadIdx.x;
//   int gridsize = blockDim.x * gridDim.x;
//   for (int ind=index; ind<n*n; ind+=gridsize) {
//     matrix[ind] = 0;
//   }
// }
// __global__ void identity_matrix(int *matrix, int n) {
//   int index = blockDim.x * blockIdx.x + threadIdx.x;
//   int gridsize = blockDim.x * gridDim.x;
//   for (int ind=index; ind<n*n; ind+=gridsize) {
//     matrix[ind] = (ind / n) == (ind % n);
//   }
// }

// // this function performs <dest> = transpose(matmul(<matrix>, transpose(<buffer>)))
// // buffer is already transposed for better performance
// __global__ void matrix_mul_transpose(int *matrix, int *dest, int *buffer, int n) {
//   int index = blockDim.x * blockIdx.x + threadIdx.x;
//   int gridsize = blockDim.x * gridDim.x;
  
//   for (int ind=index; ind<n*n; ind+=gridsize) {
//     int i = ind / n;
//     int j = ind % n;
//     int tmp = 0;
//     for (int k=0; k<n; ++k) {
//       tmp ^= matrix[j*n+k] & buffer[i*n+k];
//     }
//     dest[ind] = tmp;
//   }
// }
// // this function performs <dest> += matmul(<matrix>, transpose(<buffer>)))
// // buffer is already transposed for better performance
// __global__ void matrix_mul_add(int *matrix, int *dest, int *buffer, int n) {
//   int index = blockDim.x * blockIdx.x + threadIdx.x;
//   int gridsize = blockDim.x * gridDim.x;
  
//   for (int ind=index; ind<n*n; ind+=gridsize) {
//     int i = ind / n;
//     int j = ind % n;
//     int tmp = 0;
//     for (int k=0; k<n; ++k) {
//       tmp ^= matrix[i*n+k] & buffer[j*n+k];
//     }
//     dest[ind] ^= tmp;
//   }
// }

// // this function performs <dest> = transpose(matmul(<matrix>, transpose(<buffer>)))
// // buffer is already transposed for better performance
// __global__ void matrix_transpose(int *matrix, int *dest, int n) {
//   int index = blockDim.x * blockIdx.x + threadIdx.x;
//   int gridsize = blockDim.x * gridDim.x;
  
//   for (int ind=index; ind<n*n; ind+=gridsize) {
//     int i = ind / n;
//     int j = ind % n;
//     dest[ind] = matrix[j*n+i];
//   }
// }


// void MPI_CUDA_Calculation(const int &n,
//                           const int &m,
//                           const std::vector<std::vector<int>> &A,
//                           const std::vector<std::vector<int>> &B,
//                           std::vector<std::vector<int>> *C,
//                           const int &rank,
//                           const int &num_process,
//                           const int &number_of_block_in_a_grid,
//                           const int &number_of_thread_in_a_block) {
//   int *d_A;
//   int *d_B;
//   int *d_B_power_num_process;
//   int *d_B_power_rank_transposed;
//   int *d_C;

//   cudaMalloc((void **)&d_A, n * n * sizeof(int));
//   cudaMalloc((void **)&d_B, n * n * sizeof(int));
//   cudaMalloc((void **)&d_B_power_num_process, n * n * sizeof(int));
//   cudaMalloc((void **)&d_B_power_rank_transposed, n * n * sizeof(int));
//   cudaMalloc((void **)&d_C, n * n * sizeof(int));

//   // copy matrix to GPU
//   for (int i=0; i<n; ++i) {
//     cudaMemcpy(d_A + n * i, A[i].data(), n * sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_B + n * i, B[i].data(), n * sizeof(int), cudaMemcpyHostToDevice);
//   }
//   // calculate powers of B
//   {
//     int* cursor;
//     int* buffer;
//     // odd number needs to setup identity matrix in num_process
//     if (rank & 1) {
//       cursor = d_B_power_num_process;
//       buffer = d_B_power_rank_transposed;
//     }
//     else {
//       cursor = d_B_power_rank_transposed;
//       buffer = d_B_power_num_process;
//     }
//     // init identity
//     identity_matrix<<<number_of_block_in_a_grid, number_of_thread_in_a_block>>>(cursor, n);

//     // calculate d_B_power_rank_transposed
//     for (int t = 0; t<rank; t++) {
//       matrix_mul_transpose<<<number_of_block_in_a_grid, number_of_thread_in_a_block>>>(d_B, buffer, cursor, n);
//       int* tmp = buffer;
//       buffer = cursor;
//       cursor = tmp;
//     }
//     if (num_process < m) {
//       // calculate d_B_power_num_process
//       if ((num_process - rank) & 1) {
//         matrix_mul_transpose<<<number_of_block_in_a_grid, number_of_thread_in_a_block>>>(d_B, d_C, cursor, n);
//         cursor = d_C;
//       }
//       else {
//         matrix_mul_transpose<<<number_of_block_in_a_grid, number_of_thread_in_a_block>>>(d_B, buffer, cursor, n);
//         cursor = buffer;
//         buffer = d_C;
//       }

//       for (int t = rank+1; t<num_process; t++) {
//         matrix_mul_transpose<<<number_of_block_in_a_grid, number_of_thread_in_a_block>>>(d_B, buffer, cursor, n);
//         int* tmp = buffer;
//         buffer = cursor;
//         cursor = tmp;
//       }
//       // B = B_power_num_process transposed
//       matrix_transpose<<<number_of_block_in_a_grid, number_of_thread_in_a_block>>>(cursor, buffer, n);
//     }
//   }
//   // calculate A mat mul d_B_power_rank_transposed
//   zero_matrix<<<number_of_block_in_a_grid, number_of_thread_in_a_block>>>(d_C, n);

//   for (int t = rank; t<=m; t+=num_process) {
//     matrix_mul_add<<<number_of_block_in_a_grid, number_of_thread_in_a_block>>>(d_A, d_C, d_B_power_rank_transposed, n);
//     // increment power
//     if ((t + num_process) <= m) {
//       matrix_mul_transpose<<<number_of_block_in_a_grid, number_of_thread_in_a_block>>>(d_B_power_num_process, d_B, d_B_power_rank_transposed, n);
//       int* tmp = d_B_power_rank_transposed;
//       d_B_power_rank_transposed = d_B;
//       d_B = tmp;
//     }
//   }
//     // identity_matrix<<<number_of_block_in_a_grid, number_of_thread_in_a_block>>>(d_C, n);

//   // copy result back to CPU
//   (*C).resize(n);
//   for (int i=0; i<n; ++i) {
//     (*C)[i].resize(n);
//     cudaMemcpy((*C)[i].data(), d_C + n * i, n * sizeof(int), cudaMemcpyDeviceToHost);
//   }
  
//   cudaFree(d_A);
//   cudaFree(d_B);
//   cudaFree(d_C);
//   cudaFree(d_B_power_num_process);
//   cudaFree(d_B_power_rank_transposed);
// }

// ==============================================================
// ====    Write your functions above this line    ====
// ==============================================================
// ==============================================================

enum TASK{TEST, SAVE};

int main(int argc, char **argv) {
  int number_of_processes, rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &number_of_processes);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  double parallel_start_time;
  TASK task;
  int map_size;
  std::vector<std::vector<int>> A;
  std::vector<std::vector<int>> B;
  if (rank == 0) {
    if (argc < 3) {
      std::cout << "Error! Please use \"mpiexec -n [process number] "
                   "[--hostfile hostfile] multiple [map_size] [\"test\" or \"save\"]\"\n";
      return 1;
    } else {
      map_size = std::atoi(argv[1]);
      if (strcmp(argv[2], "test") == 0)
        task = TEST;
      else if (strcmp(argv[2], "save") == 0)
        task = SAVE;
      else {
        std::cout << "Error! Please use \"mpiexec -n [process number] "
                    "[--hostfile hostfile] multiple [map_size] [\"test\" or \"save\"]\"\n";
        return 1;
      }
    }
  }
  std::vector<std::vector<int8_t>> parallel_answer;

  if (rank == 0) {
    parallel_start_time = MPI_Wtime();
  }
  
  // ==============================================================
  // ====    Write your implementation below this line    ====
  // ==============================================================
  // ==============================================================

  // // propagate problem parameters
  // int parameters[4] = {n, m, number_of_block_in_a_grid, number_of_thread_in_a_block};
  // MPI_Bcast(parameters, 4, MPI_INT, 0, MPI_COMM_WORLD);
  // n = parameters[0];
  // m = parameters[1];
  // number_of_block_in_a_grid = parameters[2];
  // number_of_thread_in_a_block = parameters[3];
  
  // // prepare buffer for MPI transfers
  // parallel_answer.resize(n);
  // if (rank == 0) {
  //   for (int i=0; i<n; ++i) {
  //     parallel_answer[i].resize(n);
  //   }
  // }
  // else {
  //   A.resize(n);
  //   B.resize(n);
  //   for (int i=0; i<n; ++i) {
  //     A[i].resize(n);
  //     B[i].resize(n);
  //   }
  // }

  // // propagate problem array
  // for (int i=0; i<n; ++i) {
  //   MPI_Bcast(A[i].data(), n, MPI_INT, 0, MPI_COMM_WORLD);
  //   MPI_Bcast(B[i].data(), n, MPI_INT, 0, MPI_COMM_WORLD);
  // }
  
  
  // // allocate local result and calculate the results
  // std::vector<std::vector<int>> local_answer;

  // MPI_CUDA_Calculation(n, m, A, B, &local_answer, rank, number_of_processes, number_of_block_in_a_grid, number_of_thread_in_a_block);
  
  // // collect the result using bitwise-XOR
  // for (int i=0; i<n; ++i) {
  //   MPI_Reduce(local_answer[i].data(), parallel_answer[i].data(), n, MPI_INT, MPI_BXOR, 0, MPI_COMM_WORLD);
  // }

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    double parallel_end_time = MPI_Wtime();
    double parallel_running_time = parallel_end_time - parallel_start_time;
    std::cout << "parallel running time:" << parallel_running_time << std::endl;
    std::vector<std::vector<int8_t>> sequential_answer;
    double sequential_start_time = MPI_Wtime();

    sequentialWFC(map_size, &sequential_answer);

    double sequential_end_time = MPI_Wtime();
    double sequential_running_time =
        sequential_end_time - sequential_start_time;
    std::cout << "sequential running time:" << sequential_running_time
              << std::endl;
    std::cout << "speed up:" <<  sequential_running_time/parallel_running_time
              << std::endl;
    if (task == TEST) {
      if (TestAnswerCorrectness(map_size, sequential_answer)) {
        std::cout << "Correct serial solution!" << std::endl;
      }
      else {
        std::cout << "Incorrect serial solution" << std::endl;
      }
    }
    else { // SAVE heightmap in bitmap
      SaveImage(map_size, sequential_answer);
    }
 
    // for (int i=0;i<map_size;++i){
    //   for (int j=0;j<map_size;++j){
    //     std::cout << (int)sequential_answer[i][j] << " ";
    //   }
    //   std::cout << std::endl;
    // }
  }
  MPI_Finalize();
  return 0;
}