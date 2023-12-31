#pragma once
#include <sys/time.h>

#define WARMUP_NUM_CPU 2
#define EXE_NUM_CPU 2

struct CpuTimer
{
  timeval start;
  timeval stop;

  void Start()
  {
     gettimeofday(&start,NULL);
  }

  void Stop()
  {
     gettimeofday(&stop,NULL);
  }

  float Elapsed()
  {
    float elapsed;
    elapsed = (stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / 1000000.0) * 1000;
    return elapsed;
  }
};