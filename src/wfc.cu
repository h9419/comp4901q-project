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


#define MAX_MANHATTAN_DISTANCE 9
#define MAX(x,y) (((x) > (y)) ? (x) : (y))
#define MIN(x,y) (((x) < (y)) ? (x) : (y))
#define ABS(x) ((x) < 0 ? (0 - (x)) : (x))


// single core implementation
void sequentialPropagateWFC(const int &height, const int &width, const int &x, const int &y,
                  std::vector<std::vector<int>> *map,
                  std::vector<std::vector<int>> *lower_bound,
                  std::vector<std::vector<int>> *upper_bound) {
  int collapsed_value = map[0][x][y];
  // go through all cells
  for (int i=-9; i <= 9; ++i){
    for (int j=-9; j <= 9; ++j){
      // filter by manhattan distance
      int manhattan_distance = ABS(i) + ABS(j);
      if (manhattan_distance <= 9) {
        int xpos = x + i;
        int ypos = y + j;
        // map boundary
        if (xpos >= 0 && xpos < height && ypos >= 0 && ypos < width) {
          // non-determinate
          if (map[0][xpos][ypos] == -1) {
            lower_bound[0][xpos][ypos] = MAX((int)lower_bound[0][xpos][ypos], collapsed_value - manhattan_distance);
            upper_bound[0][xpos][ypos] = MIN((int)upper_bound[0][xpos][ypos], collapsed_value + manhattan_distance);
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

// single core implementation
void sequentialWFC(const int &height, const int &width, std::vector<std::vector<int>> *map) {
  std::vector<std::vector<int>> lower_bound, upper_bound;
  lower_bound = std::vector<std::vector<int>>(height, std::vector<int>(width,0));
  upper_bound = std::vector<std::vector<int>>(height, std::vector<int>(width,9));
  (*map) = std::vector<std::vector<int>>(height, std::vector<int>(width,-1));

  // big primes
  // https://stackoverflow.com/a/18994414
  const unsigned int BIG_PRIME_X = 74207281;
  const unsigned int BIG_PRIME_Y = 74207279;
  unsigned int x = rand();
  unsigned int y = rand();
  // go through the map to set all cells
  for (int i=0; i<height; ++i) {
    x = (x + BIG_PRIME_X) % height;
    for (int j=0; j<width; ++j) {
      x = (x + BIG_PRIME_X) % height;
      y = (y + BIG_PRIME_Y) % width;
      if ((*map)[x][y] == -1) {
        // set a cell randomly;
        (*map)[x][y] = lower_bound[x][y] + ABS(rand() % (1 + upper_bound[x][y] - lower_bound[x][y]));
        // propagate changes
        sequentialPropagateWFC(height, width, x, y, map, &lower_bound, &upper_bound);
      }
    }
  }
}

// single core implementation to be used with MPI
void constraintedSequentialWFC(const int height, const int width, std::vector<std::vector<int>> *map, std::vector<int> &top, std::vector<int> &bottom) {
  std::vector<std::vector<int>> lower_bound, upper_bound;
  lower_bound = std::vector<std::vector<int>>(height, std::vector<int>(width,0));
  upper_bound = std::vector<std::vector<int>>(height, std::vector<int>(width,9));
  map[0] = std::vector<std::vector<int>>(height, std::vector<int>(width,-1));
  map[0][0] = top;
  map[0][height-1] = bottom;
  
  // update according to the constraints
  for (int i=0; i<height; i += (height-1)) {
    for (int j=0; j<width; ++j) {
      sequentialPropagateWFC(height, width, i, j, map, &lower_bound, &upper_bound);
    }
  }

  // big primes
  // https://stackoverflow.com/a/18994414
  const unsigned int BIG_PRIME_X = 74207281;
  const unsigned int BIG_PRIME_Y = 74207279;
  unsigned int x = rand();
  unsigned int y = rand();
  // go through the map to set all cells
  for (int i=0; i<height; ++i) {
    x = (x + BIG_PRIME_X) % height;
    for (int j=0; j<width; ++j) {
      x = (x + BIG_PRIME_X) % height;
      y = (y + BIG_PRIME_Y) % width;
      if ((*map)[x][y] == -1) {
        // set a cell randomly;
        (*map)[x][y] = lower_bound[x][y] + ABS(rand() % (1 + upper_bound[x][y] - lower_bound[x][y]));
        // propagate changes
        sequentialPropagateWFC(height, width, x, y, map, &lower_bound, &upper_bound);
      }
    }
  }
}

// Test whether the answer follows the set of rules
bool TestAnswerCorrectness(const int &map_size, const std::vector<std::vector<int>> &answer) {
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
      if (i > 0 && ABS(answer[i][j] - answer[i-1][j]) > 1)
          return false;
      if (i < 9 && ABS(answer[i][j] - answer[i+1][j]) > 1)
          return false;
      if (j > 0 && ABS(answer[i][j] - answer[i][j-1]) > 1)
          return false;
      if (j < 9 && ABS(answer[i][j] - answer[i][j+1]) > 1)
          return false;
    }
  }
  return true;
}

// This method saves the table to "saved_result.bmp" in your current directory
// the allows the generated map to be exported into other programs
void SaveImage(const int &map_size, const std::vector<std::vector<int>> &answer) {
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

// set initial values for allocated memory
__global__ void cudaInitMemory(int* d_map, int* d_lower_bound, int* d_upper_bound, int size) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x;
        i < size;
        i += blockDim.x * gridDim.x)
  {
    d_map[i] = -1;
    d_lower_bound[i] = 0;
    d_upper_bound[i] = 9;
  }
}

// kernel that recieves a random number to determin collapsed value and propagate changes
// threads in MAX_MANHATTAN_DISTANCE by MAX_MANHATTAN_DISTANCE
__global__ void randomPropagateWFC(const int width,
                                  const int height,
                                  const int x,
                                  const int y,
                                  const unsigned int random,
                                  int* map,
                                  int* lower_bound,
                                  int* upper_bound)
{
  if (map[x * width + y] != -1)
    return;
  int collapsed_value = lower_bound[x * width + y] + random % (upper_bound[x * width + y] - lower_bound[x * width + y] + 1);
  int i = threadIdx.x - MAX_MANHATTAN_DISTANCE;
  int j = threadIdx.y - MAX_MANHATTAN_DISTANCE;
  int manhattan_distance = ABS(i) + ABS(j);
  if (manhattan_distance <= MAX_MANHATTAN_DISTANCE) {
    int xpos = x + i;
    int ypos = y + j;
    // map boundary
    if (xpos >= 0 && xpos < height && ypos >= 0 && ypos < width) {
      // non-determinate
      if (map[xpos * width + ypos] == -1) {
        lower_bound[xpos * width + ypos] = MAX((int)lower_bound[xpos * width + ypos], collapsed_value - manhattan_distance);
        upper_bound[xpos * width + ypos] = MIN((int)upper_bound[xpos * width + ypos], collapsed_value + manhattan_distance);
        // only 1 state left, set state
        if (lower_bound[xpos * width + ypos] == upper_bound[xpos * width + ypos]) {
          map[xpos * width + ypos] = upper_bound[xpos * width + ypos];
        }
      }
    }
  }
}

// kernel that recieves the collapsed value and propagate changes
// threads in MAX_MANHATTAN_DISTANCE by MAX_MANHATTAN_DISTANCE
__global__ void setAndPropagateWFC(const int width,
                                  const int height,
                                  const int x,
                                  const int y,
                                  const int collapsed_value,
                                  int* map,
                                  int* lower_bound,
                                  int* upper_bound) {
  int i = threadIdx.x - MAX_MANHATTAN_DISTANCE;
  int j = threadIdx.y - MAX_MANHATTAN_DISTANCE;
  int manhattan_distance = ABS(i) + ABS(j);
  if (manhattan_distance <= MAX_MANHATTAN_DISTANCE) {
    int xpos = x + i;
    int ypos = y + j;
    // map boundary
    if (xpos >= 0 && xpos < height && ypos >= 0 && ypos < width) {
      // non-determinate
      if (map[xpos * width + ypos] == -1) {
        lower_bound[xpos * width + ypos] = MAX((int)lower_bound[xpos * width + ypos], collapsed_value - manhattan_distance);
        upper_bound[xpos * width + ypos] = MIN((int)upper_bound[xpos * width + ypos], collapsed_value + manhattan_distance);
        // only 1 state left, set state
        if (lower_bound[xpos * width + ypos] == upper_bound[xpos * width + ypos]) {
          map[xpos * width + ypos] = upper_bound[xpos * width + ypos];
        }
      }
    }
  }
}

// this method is single process, single GPU version of the wave function collapse algorithm
void cudaWFC(const int height, const int width, std::vector<std::vector<int>> *map) {
  const dim3 threads(MAX_MANHATTAN_DISTANCE * 2 + 1, MAX_MANHATTAN_DISTANCE * 2 + 1, 1);
  int *d_map;
  int *d_lower_bound;
  int *d_upper_bound;
  cudaMalloc((void **)&d_map,         height * width * sizeof(int));
  cudaMalloc((void **)&d_lower_bound, height * width * sizeof(int));
  cudaMalloc((void **)&d_upper_bound, height * width * sizeof(int));
  cudaInitMemory<<<256,256>>>(d_map, d_lower_bound, d_upper_bound, height * width);
  
  const unsigned int BIG_PRIME_X = 74207281;
  const unsigned int BIG_PRIME_Y = 74207279;
  unsigned int x = rand();
  unsigned int y = rand();
  // go through the map to set all cells
  for (int i=0; i<height; ++i) {
    x = (x + BIG_PRIME_X) % height;
    for (int j=0; j<width; ++j) {
      x = (x + BIG_PRIME_X) % height;
      y = (y + BIG_PRIME_Y) % width;
      // set a cell randomly and propagate changes
      randomPropagateWFC<<<1, threads>>>(width, height, x, y, rand(),
                                      d_map, d_lower_bound, d_upper_bound);
    }
  }


  map[0].resize(height);
  for (int i=0; i<height; ++i) {
    map[0][i].resize(width);
    cudaMemcpy(map[0][i].data(), d_map + width * i, width * sizeof(int), cudaMemcpyDeviceToHost);
  }
  cudaFree(d_map);
  cudaFree(d_lower_bound);
  cudaFree(d_upper_bound);
}

// this is the single GPU version of the wave function collapse algorithm to be used inside MPI
void constraintedCudaWFC(const int height, const int width, std::vector<std::vector<int>> *map, std::vector<int> &top, std::vector<int> &bottom) {
  const dim3 threads(MAX_MANHATTAN_DISTANCE * 2 + 1, MAX_MANHATTAN_DISTANCE * 2 + 1, 1);
  int *d_map;
  int *d_lower_bound;
  int *d_upper_bound;
  cudaMalloc((void **)&d_map,         height * width * sizeof(int));
  cudaMalloc((void **)&d_lower_bound, height * width * sizeof(int));
  cudaMalloc((void **)&d_upper_bound, height * width * sizeof(int));
  cudaInitMemory<<<256,256>>>(d_map, d_lower_bound, d_upper_bound, height * width);
  for (int j=0; j<width; ++j) {
    setAndPropagateWFC<<<1, threads>>>(width, height, 0, j, top[j],
                                      d_map, d_lower_bound, d_upper_bound);
    setAndPropagateWFC<<<1, threads>>>(width, height, height-1, j, bottom[j],
                                      d_map, d_lower_bound, d_upper_bound);
  }
  
  const unsigned int BIG_PRIME_X = 74207281;
  const unsigned int BIG_PRIME_Y = 74207279;
  unsigned int x = rand();
  unsigned int y = rand();
  // go through the map to set all cells
  for (int i=0; i<height; ++i) {
    x = (x + BIG_PRIME_X) % height;
    for (int j=0; j<width; ++j) {
      x = (x + BIG_PRIME_X) % height;
      y = (y + BIG_PRIME_Y) % width;
      // set a cell randomly and propagate changes
      randomPropagateWFC<<<1, threads>>>(width, height, x, y, rand(),
                                      d_map, d_lower_bound, d_upper_bound);
    }
  }


  map[0].resize(height);
  for (int i=0; i<height; ++i) {
    map[0][i].resize(width);
    cudaMemcpy(map[0][i].data(), d_map + width * i, width * sizeof(int), cudaMemcpyDeviceToHost);
  }
  cudaFree(d_map);
  cudaFree(d_lower_bound);
  cudaFree(d_upper_bound);
}

// this variable store the command parsed
enum TASK{TEST_MPI, TEST_CUDA, TEST_MPI_CUDA, SAVE};

int main(int argc, char **argv) {
  int number_of_processes, rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &number_of_processes);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  double parallel_start_time;
  srand(time(NULL) + rank);
  TASK task;
  int map_size;
  std::vector<std::vector<int>> A;
  std::vector<std::vector<int>> B;
  if (rank == 0) {
    if (argc < 3) {
      std::cout << "Error! Please use \"mpiexec -n [process number] "
                   "[--hostfile hostfile] multiple [map_size] [\"test_mpi\", \"test_cuda\", \"test_mpi_cuda\" or \"save\"]\"\n";
      return 1;
    } else {
      map_size = std::atoi(argv[1]);
      if (strcmp(argv[2], "test_mpi") == 0)
        task = TEST_MPI;
      else if (strcmp(argv[2], "test_cuda") == 0)
        task = TEST_CUDA;
      else if (strcmp(argv[2], "test_mpi_cuda") == 0)
        task = TEST_MPI_CUDA;
      else if (strcmp(argv[2], "save") == 0)
        task = SAVE;
      else {
        std::cout << "Error! Please use \"mpiexec -n [process number] "
                    "[--hostfile hostfile] multiple [map_size] [\"test_mpi\", \"test_cuda\", \"test_mpi_cuda\" or \"save\"]\"\n";
        return 1;
      }
    }
  }
  std::vector<std::vector<int>> parallel_answer;

  MPI_Bcast(&task, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (task == TEST_MPI || task == TEST_MPI_CUDA || task == SAVE) {
    
    if (rank == 0) {
      parallel_start_time = MPI_Wtime();
    }
    
    // ==============================================================
    // ====               MPI implementation below               ====
    // ==============================================================
    // ==============================================================
    
    MPI_Bcast(&map_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int row_per_process = MAX(9, (map_size + number_of_processes - 1) / number_of_processes);

    // generate border edge to pass around
    std::vector<int> top_border, bottom_border;
    std::vector<std::vector<int>> local_answer;
    top_border.resize(map_size);
    bottom_border.resize(map_size);
    top_border[0] = rand() % 10;
    for (int i=1; i<map_size; ++i) {
      if (top_border[i-1] == 0)
        top_border[i] = top_border[i-1] + (rand() % 2);
      else if (top_border[i-1] == 9)
        top_border[i] = top_border[i-1] + (rand() % 2) - 1;
      else 
        top_border[i] = top_border[i-1] + (rand() % 3) - 1;
    }
    // send the edge around in a ring
    MPI_Request sendRequest, recvRequest;
    MPI_Isend(top_border.data(), map_size, MPI_INT, (rank + number_of_processes - 1) % number_of_processes, 0, MPI_COMM_WORLD, &sendRequest);
    MPI_Irecv(bottom_border.data(), map_size, MPI_INT, (rank + 1) % number_of_processes, 0, MPI_COMM_WORLD, &recvRequest);
    MPI_Wait(&sendRequest, MPI_STATUS_IGNORE);
    MPI_Wait(&recvRequest, MPI_STATUS_IGNORE);
    // MPI-only version without GPU acceleration
    if (task == TEST_MPI)
      constraintedSequentialWFC(row_per_process, map_size, &local_answer, top_border, bottom_border);
    // MPI+CUDA version with GPU acceleration
    else
      constraintedCudaWFC(row_per_process, map_size, &local_answer, top_border, bottom_border);

    if (rank == 0) {
      parallel_answer = local_answer;
      parallel_answer.resize(map_size);
      for (int h = row_per_process; h < map_size; ++h) {
        parallel_answer[h].resize(map_size);
        MPI_Recv(parallel_answer[h].data(), map_size, MPI_INT, h / row_per_process, 0, MPI_COMM_WORLD, NULL);
      }
    }
    else {
      int start = row_per_process * rank;
      int iter = MIN(row_per_process * (rank+1), map_size) - start;
      for (int h = 0; h < iter; ++h) {
        MPI_Send(local_answer[h].data(), map_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
      }
    }
  }
  
  // ==============================================================
  // ====               MPI implementation above               ====
  // ==============================================================
  // ==============================================================

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    double parallel_end_time = MPI_Wtime();
    // single node testing single GPU
    if (task == TEST_CUDA) {
      parallel_start_time = MPI_Wtime();
      cudaWFC(map_size, map_size, &parallel_answer);
      parallel_end_time = MPI_Wtime();
    }
    
    double parallel_running_time = parallel_end_time - parallel_start_time;
    if (task == TEST_CUDA)
      std::cout << "CUDA";
    else if (task == TEST_MPI)
      std::cout << "MPI";
    else
      std::cout << "MPI+CUDA";
    std::cout << " parallel running time:" << parallel_running_time << std::endl;
    if (TestAnswerCorrectness(map_size, parallel_answer)) {
        std::cout << "Valid parallel solution!" << std::endl;
      }
      else {
        std::cout << "Invalid parallel solution" << std::endl;
      }
    if (task == SAVE) {
      std::cout << "Image saved to \"saved_result.bmp\"" << std::endl;
      SaveImage(map_size, parallel_answer);
    }
    else {
      
      std::vector<std::vector<int>> sequential_answer;
      double sequential_start_time = MPI_Wtime();

      sequentialWFC(map_size, map_size, &sequential_answer);

      double sequential_end_time = MPI_Wtime();
      double sequential_running_time =
          sequential_end_time - sequential_start_time;
      std::cout << "sequential running time:" << sequential_running_time
                << std::endl;
      std::cout << "speed up:" <<  sequential_running_time/parallel_running_time
                << std::endl;
      
      if (TestAnswerCorrectness(map_size, sequential_answer)) {
        std::cout << "Valid serial solution!" << std::endl;
      }
      else {
        std::cout << "Invalid serial solution" << std::endl;
      }
    }
  }
  MPI_Finalize();
  return 0;
}