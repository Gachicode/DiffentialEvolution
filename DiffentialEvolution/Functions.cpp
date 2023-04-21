#include "Functions.h"

#include <cmath>
#include <format>
#include <iostream>

// Eggholder function declaration
double eggholder_func(double x, double y)
{
    return -(y + 47.0) * sin(sqrt(abs(y + x * 0.5 + 47.0))) - x * sin(sqrt(abs(x - (y + 47.0))));
}

// Ackley function declaration
double ackley_func(double x, double y)
{
    return -20.0 * exp(-0.2 * sqrt(0.5 * (x * x + y * y))) - exp(0.5 * (cos(2.0 * PI * x) + cos(2.0 * PI * y))) + exp(1.0) + 20.0;
}

// Rosenbrock function declaration
double rosenbrock_func(double x, double y)
{
    return (x - 1) * (x - 1) + 100 * (y - x * x) * (y - x * x);
}

// Basic func
double func(double x, double y)
{
    extern int FUNC_TYPE;

    switch (FUNC_TYPE)
    {
    case 0:
        return rosenbrock_func(x, y);
    case 1:
        return ackley_func(x, y);
    default:
        return eggholder_func(x, y);
    }
}

double evaluate(int D, double tmp[], long* nfeval)
{
    int i;
    double result = 0;

    (*nfeval)++;
    for (i = 0; i < D - 1; i++)
        result += func(tmp[i], tmp[i + 1]);
    
    extern int PRINT_VALUES;

    if (PRINT_VALUES)
        std::cout << std::format("{:2} {:.4}\n", i, result);

    return result;
}

void copy_vector(double a[], double b[])
{
    for (int k = 0; k < MAX_GENS; k++)
        a[k] = b[k];
}

void copy_array(double dest[MAX_POPULATION][MAX_GENS], double source[MAX_POPULATION][MAX_GENS])
{
    for (int j = 0; j < MAX_POPULATION; j++)
        for (int k = 0; k < MAX_GENS; k++)
            dest[j][k] = source[j][k];
}

double check_bounds(double value, double upper, double lower) {
    if (value > upper)
        return upper;
    else if (value < lower)
        return lower;
    else
        return value;
};