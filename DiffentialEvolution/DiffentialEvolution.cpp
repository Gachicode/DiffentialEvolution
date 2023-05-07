#include <chrono>
#include <cmath>
#include <fstream>
#include <matplot/matplot.h>
#include <string>
#include <random>
#include <vector>
#include <variant>
#include "Functions.h"
#include <math.h>

int BUILD_PLOT = 0;
int FUNC_TYPE = -1;
int PRINT_VALUES = 0;
int PRINT_RESULTS = 0;

double c[MAX_POPULATION][MAX_GENS], d[MAX_POPULATION][MAX_GENS];
double oldarray[MAX_POPULATION][MAX_GENS];
double newarray[MAX_POPULATION][MAX_GENS];
double swaparray[MAX_POPULATION][MAX_GENS];

std::vector<double> best_points_x;
std::vector<double> best_points_y;
std::vector<double> best_points_z;

std::vector<double> general_main(int strategy, int genmax, int D, int NP, double inibound_h, double inibound_l, double F, double CR)
{
    int   i, j, L, n;      // counting variables                 
    int   r1, r2, r3, r4;  //   
    int   r5;              // placeholders for random indexes        
    int   imin;            // index to member with lowest energy    
    int   gen;             // generation counter
    long  nfeval;          // number of function evaluations     
    double trial_energy;    // buffer variable                    
    double  tmp[MAX_GENS] = { 0 },
        best[MAX_GENS] = { 0 },
        bestit[MAX_GENS] = { 0 }; // members  
    double energy[MAX_POPULATION] = { 0 };  // obj. funct. values                 
    double emin;            // help variables                     

    // Uniform real distribution
    std::random_device rand;
    std::mt19937 engine(rand()); // A Mersenne Twister pseudo-random generator of 32-bit numbers with a state size of 19937 bits.
    std::uniform_real_distribution<double> dist(LEFT, RIGHT);

    nfeval = 0;           // reset number of function evaluations 
    double r;

    // Initialization
    // Right now this part is kept fairly simple and just generates
    // random numbers in the range [-initfac, +initfac]. You might
    // want to extend the init part such that you can initialize
    // each parameter separately.

    // spread initial population members
    for (i = 0; i < NP; i++)
    {
        for (j = 0; j < D; j++)
        {
            r = dist(engine);
            c[i][j] = inibound_l + r * (inibound_h - inibound_l);
        }
        energy[i] = evaluate(D, c[i], &nfeval);
    }

    emin = energy[0];
    imin = 0;
    for (i = 1; i < NP; i++)
    {
        if (energy[i] < emin)
        {
            emin = energy[i];
            imin = i;
        }
    }

    copy_vector(best, c[imin]);
    copy_vector(bestit, c[imin]);

    // old population (generation G)
    copy_array(oldarray, c);
    // new population (generation G+1)
    copy_array(newarray, d);

    auto start = std::chrono::system_clock::now();

    // Iteration loop
    gen = 1; // generation counter reset
    while ((gen < genmax))
    {
        gen++;
        imin = 0;

        for (i = 0; i < NP; i++)
        {
            // Pick a random population member 
            do {
                // Endless loop for NP < 2 !!!
                r = dist(engine);
                r1 = (int)(r * NP);
            } while (r1 == i);

            do {
                // Endless loop for NP < 3 !!!
                r = dist(engine);
                r2 = (int)(r * NP);
            } while ((r2 == i) || (r2 == r1));

            do {
                // Endless loop for NP < 4 !!!     
                r3 = (int)(dist(engine) * NP);
            } while ((r3 == i) || (r3 == r1) || (r3 == r2));

            do {
                // Endless loop for NP < 5 !!!     
                r4 = (int)(dist(engine) * NP);
            } while ((r4 == i) || (r4 == r1) || (r4 == r2) || (r4 == r3));

            do {
                // Endless loop for NP < 6 !!!     
                r5 = (int)(dist(engine) * NP);
            } while ((r5 == i) || (r5 == r1) || (r5 == r2) || (r5 == r3) || (r5 == r4));

            // Choice of strategy
            // We have tried to come up with a sensible naming-convention: DE/x/y/z
            //   DE :  stands for Differential Evolution
            //   x  :  a string which denotes the vector to be perturbed
            //   y  :  number of difference vectors taken for perturbation of x
            //   z  :  crossover method (exp = exponential, bin = binomial)
            //
            // There are some simple rules which are worth following:
            //   1)  F is usually between 0.5 and 1 (in rare cases > 1)
            //   2)  CR is between 0 and 1 with 0., 0.3, 0.7 and 1. being worth to be tried first
            //   3)  To start off NP = 10*D is a reasonable choice. Increase NP if misconvergence=
            //       happens.
            //   4)  If you increase NP, F usually has to be decreased
            //   5)  When the DE/best... schemes fail DE/rand... usually works and vice versa


            // EXPONENTIAL CROSSOVER
            // DE/best/1/exp
            // Our oldest strategy but still not bad. However, we have found several
            // optimization problems where misconvergence occurs.

            // strategy DE0 (not in our paper)
            if (strategy == 2) //based
            {
                for (int k = 0; k < MAX_GENS; k++)
                    tmp[k] = oldarray[i][k];

                n = (int)(dist(engine) * D);
                L = 0;
                do {
                    tmp[n] = bestit[n] + F * (oldarray[r2][n] - oldarray[r3][n]);
                    tmp[n] = check_bounds(tmp[n], inibound_h, inibound_l);
                    n = (n + 1) % D;
                    L++;
                } while ((dist(engine) < CR) && (L < D));
            }
            // DE/rand/1/exp
            // This is one of my favourite strategies. It works especially well when the
            // "bestit[]"-schemes experience misconvergence. Try e.g. F=0.7 and CR = 0.5
            // as a first guess.
            // strategy DE1 in the techreport
            else if (strategy == 1) //mod
            {
                for (int k = 0; k < MAX_GENS; k++)
                    tmp[k] = oldarray[i][k];

                n = (int)(dist(engine) * D);
                L = 0;
                do {
                    tmp[n] = oldarray[r1][n] + F * (oldarray[r2][n] - oldarray[r3][n]);
                    tmp[n] = check_bounds(tmp[n], inibound_h, inibound_l);
                    n = (n + 1) % D;
                    L++;
                } while ((dist(engine) < CR) && (L < D));
            }

            // DE/best/2/bin
            else if (strategy == 3) //mod 4 parents
            {
                for (int k = 0; k < MAX_GENS; k++)
                    tmp[k] = oldarray[i][k];

                n = (int)(dist(engine) * D);
                for (L = 0; L < D; L++)
                {
                    if ((dist(engine) < CR) || L == (D - 1))
                    {
                        tmp[n] = bestit[n] + (oldarray[r1][n] + oldarray[r2][n] - oldarray[r3][n] - oldarray[r4][n]) * F;
                        tmp[n] = check_bounds(tmp[n], inibound_h, inibound_l);
                    }

                    n = (n + 1) % D;
                }
            }


            // Trial mutation now in tmp[]. Test how good this choice really was.
            trial_energy = evaluate(D, tmp, &nfeval);  // Evaluate new vector in tmp[]
            // improved objective function value?
            if (trial_energy <= energy[i]) {
                energy[i] = trial_energy;
                for (int k = 0; k < MAX_GENS; k++) {
                    newarray[i][k] = tmp[k];
                }
                // Was this a new minimum?
                if (trial_energy < emin) {
                    // reset emin to new low...
                    emin = trial_energy;
                    imin = i;
                    for (int k = 0; k < MAX_GENS; k++) {
                        best[k] = tmp[k];
                    }
                }
            }
            else {
                // replace target with old value
                for (int k = 0; k < MAX_GENS; k++) {
                    newarray[i][k] = oldarray[i][k];
                }
            }
        }

        best_points_x.push_back(best[0]);
        best_points_y.push_back(best[1]);
        best_points_z.push_back(emin);

        copy_vector(bestit, best);  // Save best population member of current iteration

        // swap population arrays. New generation becomes old one
        copy_array(swaparray, oldarray);
        copy_array(oldarray, newarray);
        copy_array(newarray, swaparray);
    }
    auto end = std::chrono::system_clock::now();

    if (PRINT_RESULTS)
    {
        std::cout << std::format("\n\nBest-so-far obj. funct. value: {:.10f}\n", emin);
        for (j = 0; j < D; j++)
            std::cout << std::format("best[{}]: {:.7f}\n", j, best[j]);

        std::cout << std::format("Generation: {}  NFEs: {}\n", gen, nfeval);
        std::cout << std::format("Strategy: {}  NP: {}  F: {}  CR: {}\n", strategy, NP, F, CR);
        std::cout << std::format("Elapsed time: {}\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start));
    }

    std::vector<double> total_result;

    for (int i = 0; i < D; i++)
        total_result.push_back(best[i]);

    total_result.push_back(emin);
    total_result.push_back((double)strategy);
    total_result.push_back((double)NP);
    total_result.push_back(F);
    total_result.push_back(CR);

    return total_result;
};

int main(int argc, char* argv[]) {
    for (int i = 0; i < argc; i++)
    {
        if (strcmp(argv[i], "--no-plot") == 0)
            BUILD_PLOT = 0;
        else if (strcmp(argv[i], "--no-values") == 0)
            PRINT_VALUES = 0;
        else if (strcmp(argv[i], "--func") == 0)
        {
            if (i + 1 >= argc)
            {
                std::cout << "Unknown function code! Eggholder will be used!\n";
                FUNC_TYPE = -1;
            }
            else
                FUNC_TYPE = atoi(argv[i + 1]);
        }
    }

    //std::ifstream stream("demo.dat");
    //if (!stream.is_open())
    //{
    //    std::cout << "The file demo.dat was not opened\n";
    //    exit(1);
    //}
    //stream >> strategy;       // choice of strategy
    //stream >> genmax;         // maximum number of generations
    //stream >> D;              // number of parameters
    //stream >> NP;             // population size.
    //stream >> inibound_h;     // upper parameter bound for init
    //stream >> inibound_l;     // lower parameter bound for init
    //stream >> F;              // weight factor
    //stream >> CR;             // crossing over factor
    //stream.close();

    const int D = 4; //num of genes
    double egg_holder_optimum = -915.61991 * D + 862.10466;
    double acceptable_error = 0.05; //if error equals 1% or less -> count as converged
    int repeats = 250; //repeat 

    double F, CR;
    int iters = 200;
    F = 1; //We get num of parents by multiplication amount of genes by this number
    CR = 0.5;
    int NP;

    std::ofstream myfile;
    myfile.open("example.csv");
    myfile << "F; strategy 1 convergence; strategy 1 MAPE; strategy 2 convergence; strategy 2 MAPE; strategy 3 convergence; strategy 3 MAPE\n";
    // all conv and MAPE in %
    //strat 1 - base
    //strat 2 - our mod 
    //strat 3 - our mod with 4 parents
    for (NP = D*2; NP <= 20*D; NP++)
    {
        std::string total_string = "";
        //for(int iters = 10; iters <= 500; iters +=10)
        //{
        total_string = std::to_string(NP) + ";";
        //std::vector<std::vector<double>> summary;
        for (size_t strategy = 1; strategy <= 3; strategy++)
        {
            //summary.clear();
            int converged = 0, counter = 0;
            double MAPE = 0;
            //summary.push_back(general_main(strategy, iters, D, D * NP_multi, UPPER_BOUND, LOWER_BOUND, 1, 0.5));


            //for (double F = 0; F <= 2; F += 0.2)
            for (int a = 0; a < repeats; a++)
            {
                counter++;
                //summary.push_back(general_main(strategy, iters, D, D * NP_multi, UPPER_BOUND, LOWER_BOUND, F, CR));
                double func_result = general_main(strategy, iters, D, NP, UPPER_BOUND, LOWER_BOUND, F, CR)[D];
                MAPE += fabs(egg_holder_optimum - func_result);
                //std::cout << summary.back()[D] << '\n';
                if (fabs(func_result) > fabs((1 - acceptable_error) * egg_holder_optimum)) // check if value in right interval (95%)
                    converged++;

            }
            double convergeance_rate = (double)converged / (double)counter;


            /*        std::sort(
                        summary.begin(),
                        summary.end(),
                        [](auto& left, auto& right) { return left[D] < right[D]; }
                    );*/


                    //std::cout << std::format("\n\nStrategy: {}\n", summary[0][D + 1]);
                    //for (size_t i = 0; i < 3; i++)
                    //{
                    //    std::cout << std::format("\nBest {} obj. funct. value: {:.10f}", i + 1, summary[i][D]);
                    //    //for (int j = 0; j < D; j++)
                    //    //    std::cout << std::format("\nbest[{}]: {:.7f}", j, summary[i][j]);
                    //    //std::cout << std::format("\nNP: {}  F: {:.1f}  CR: {:.1f}\n", summary[i][D + 2], summary[i][D + 3], summary[i][D + 4]);

                    //}
            MAPE = MAPE / counter / fabs(egg_holder_optimum);
            //std::cout << std::format("\nConverged: {:2f} %", convergeance_rate * 100)
            //    << "\nMAPE: " << MAPE * 100 << '%';
            total_string = total_string + std::to_string(convergeance_rate * 100) + ";" + std::to_string(MAPE * 100) + ";";
            //}
        }
        myfile << total_string << '\n';
        std::cout << "NP=" << NP << " iteration with 3 strats is over\n";
    }
    myfile.close();

    return 0;
}
