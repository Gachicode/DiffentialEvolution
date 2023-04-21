#pragma once

#include "Constants.h"

double eggholder_func(double x, double y);
double ackley_func(double x, double y);
double rosenbrock_func(double x, double y);

double func(double x, double y);
double evaluate(int D, double tmp[], long* nfeval);
double check_bounds(double value, double upper, double lower);

void copy_vector(double a[], double b[]);
void copy_array(double dest[MAX_POPULATION][MAX_GENS], double source[MAX_POPULATION][MAX_GENS]);