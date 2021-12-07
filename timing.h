#ifndef TIMING_H
#define TIMING_H

#include <sys/time.h>
#include <iostream>
//#include <chrono>
//#include <ctime>  


/* Subtract the `struct timeval' value 'then' from 'now',
   returning the difference as a float representing seconds
   elapsed.
*/
float elapsedTime(struct timeval now, struct timeval then);

float currentTime();

//Takes the GPU time and CPU time, compares, and prints.
void printTimes(double gTimeCost, double cTimeCost);

#endif
