#include "ConfigFile.tpp"
#include <chrono>
#include <cmath>
#include <complex> // Pour les nombres complexes
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <valarray>
#include <omp.h>

using namespace std;
typedef complex<double> cdouble;
typedef valarray<cdouble> vec_cmplx;

constexpr double PI = 3.1415926535897932384626433832795028841971e0;
#define NTHREADS 3

#define IN(x, a, b) (a <= (x) && (x) <= b)
#define SQ(x) ((x)*(x))
#define NORM2(z) (conj(z)*(z))
#define FOR(i, arr) for(size_t i=0;i<arr.size();++i)

// Fonction resolvant le systeme d'equations A * solution = rhs
// où A est une matrice tridiagonale
template<class T>
void triangular_solve(valarray<T> const& diag,  valarray<T> const& lower, valarray<T> const& upper,
                 valarray<T> const& rhs, valarray<T>& solution)
{
    valarray<T> new_diag = diag;
    valarray<T> new_rhs = rhs;

    // forward elimination
    for (int i(1); i < diag.size(); ++i) {
        T pivot = lower[i - 1] / new_diag[i - 1];
        new_diag[i] -= pivot * upper[i - 1];
        new_rhs[i] -= pivot * new_rhs[i - 1];
    }

    solution.resize(diag.size());

    // solve last equation
    solution[diag.size() - 1] = new_rhs[diag.size() - 1] / new_diag[diag.size() - 1];

    // backward substitution
    for (int i = diag.size() - 2; i >= 0; --i) {
        solution[i] = (new_rhs[i] - upper[i] * solution[i + 1]) / new_diag[i];
    }
}

// Calcule la moyenne d'un observable Op
cdouble compute_mean(const vec_cmplx& psi, const vec_cmplx& aOp, const vec_cmplx& dOp, const vec_cmplx& cOp, double dx){
    vec_cmplx tmp(cdouble(0,0), psi.size());

    FOR(i, aOp){
        tmp[i] += aOp[i]*psi[i+1];
    }
    FOR(i, dOp){
        tmp[i] += dOp[i]*psi[i];
    }
    FOR(i, cOp){
        tmp[i+1] += cOp[i]*psi[i];
    }
    cdouble out = 0;
    #pragma omp parallel for num_threads(NTHREADS)
    FOR(i, tmp){
        out += conj(psi[i])*tmp[i];
    }
    return out * dx;
}


// TODO Potentiel V(x) :
double V(double x, double V0, double om0, double xa, double xb, double xL, double xR, double m=1.0)
{
    double out=0;
    if(IN(x, xL, xa)){
        out = 0.5*m*SQ(om0)*SQ((x-xa)/(xL-xa));
    }
    else if(IN(x, xa, xb)){
        out = V0*SQ( sin(PI*(x-xa)/(xb-xa)) );
    }
    else if(IN(x, xb, xR)){
        out = 0.5*m*SQ(om0)*SQ((x-xb)/(xR-xb));
    }
    return out;
}

// Declaration des diagnostiques de la particule d'apres sa fonction d'onde psi :
//  - prob: calcule la probabilite de trouver la particule dans un intervalle [x_i, x_j]
//  - E:    calcule son energie,
//  - xmoy: calcule sa position moyenne,
//  - x2moy:calcule sa position au carre moyenne,
//  - pmoy: calcule sa quantite de mouvement moyenne,
//  - p2moy:calcule sa quantite de mouvement au carre moyenne.


// TODO: calculer la probabilite de trouver la particule dans un intervalle [x_i, x_j]
double prob(const vec_cmplx& psi, double dx, size_t start_idx, size_t end_idx)
{
    cdouble out=0;
    #pragma omp parallel for num_threads(NTHREADS)
    for(size_t i=start_idx; i<end_idx; ++i){
        out += 0.5*(NORM2(psi[i])+NORM2(psi[i+1]))*dx;
    }
    return out.real();
}

// TODO calculer l'energie
double E(const vec_cmplx& psi, const vec_cmplx& aH, const vec_cmplx& dH, const vec_cmplx& cH, double dx){
    return compute_mean(psi, aH, dH, cH, dx).real();
}

// TODO calculer xmoyenne
double xmoy(const vec_cmplx& psi, const vec_cmplx& x, const vec_cmplx& zeros, double dx)
{
    return compute_mean(psi, zeros, x, zeros, dx).real();
}

// TODO calculer x.^2 moyenne
double x2moy(const vec_cmplx& psi, const vec_cmplx& x2, const vec_cmplx& zeros, double dx)
{
    return compute_mean(psi, zeros, x2, zeros, dx).real();
}

// TODO calculer p moyenne
double pmoy(const vec_cmplx& psi, const vec_cmplx& aP, const vec_cmplx& dP, const vec_cmplx& cP, double dx)
{
    return compute_mean(psi, aP, dP, cP, dx).real();
}

// TODO calculer p.^2 moyenne
double p2moy(const vec_cmplx& psi, const vec_cmplx& aP2, const vec_cmplx& dP2, const vec_cmplx& cP2, double dx)
{
    return compute_mean(psi, aP2, dP2, cP2, dx).real();
}

// TODO calculer la normalization
vec_cmplx normalize(vec_cmplx const& psi, double const& dx)
{
    cdouble integral=0;
    for(size_t i=0; i<psi.size()-1; ++i){
        integral += dx*0.5*(NORM2(psi[i])+NORM2(psi[i+1]));
    }
    return psi/integral.real();
}

void writePsi(ofstream& ofs, const cdouble& psi){
    double out[] = {abs(psi), real(psi), imag(psi)};
    ofs.write((char*)out, sizeof(double)*3);
}


void writeObs(ofstream& ofs, double t, double prob1, double prob2, double energy, double avg_x, double avg_x2, double avg_p, double avg_p2){
    double out[] = {t, prob1, prob2, energy, avg_x, avg_x2, avg_p, avg_p2};
    ofs.write((char*)out, sizeof(double)*8);
}



int
main(int argc, char** argv)
{
    cout << "Max. usable threads: " << omp_get_max_threads() << endl;

    complex<double> complex_i = complex<double>(0, 1); // Nombre imaginaire i

    string inputPath("configuration.in"); // Fichier d'input par defaut
    if (argc > 1) // Fichier d'input specifie par l'utilisateur ("./Exercice8 config_perso.in")
        inputPath = argv[1];

    ConfigFile configFile(
      inputPath); // Les parametres sont lus et stockes dans une "map" de strings.

    for (int i(2); i < argc;
         ++i) // Input complementaires ("./Exercice8 config_perso.in input_scan=[valeur]")
        configFile.process(argv[i]);

    // Set verbosity level. Set to 0 to reduce printouts in console.
    const int verbose = configFile.get<int>("verbose");
    configFile.setVerbosity(verbose);

    // Parametres physiques :
    double hbar = 1.;
    double m = 1.;
    double tfin = configFile.get<double>("tfin");
    double xL = configFile.get<double>("xL");
    double xR = configFile.get<double>("xR");
    double xa = configFile.get<double>("xa");
    double xb = configFile.get<double>("xb");
    double V0 = configFile.get<double>("V0");
    double om0 = configFile.get<double>("om0");
    double n  = configFile.get<int>("n"); // Read mode number as integer, convert to double

    double x0 = configFile.get<double>("x0");
    double sigma0 = configFile.get<double>("sigma_norm") * (xR - xL);

    int Nsteps = configFile.get<int>("Nsteps");
    int Nintervals = configFile.get<int>("Nintervals");
    
    // TODO: initialiser le paquet d'onde, equation (4.116) du cours
    double L = xR - xL;
    double k0 = 2*PI*n/L;

    int Npoints = Nintervals + 1;
    double dx = (xR - xL) / Nintervals;
    double dt = tfin / Nsteps;
    
    size_t last = Nintervals-1;
    size_t last_diag = Npoints-1;


    cdouble j(0.0, 1.0);

    const auto simulationStart = std::chrono::steady_clock::now();

    // Maillage :
    valarray<double> x((double)0,Npoints);
    vec_cmplx cx(cdouble(0,0),Npoints);
    for (int i=0; i < Npoints; ++i){
        x[i] = xL + i * dx;
        cx[i] = x[i];
    }

    
    vec_cmplx zeros(cdouble(0,0), Nintervals);

    valarray<double> x2((double)0, x.size());
    vec_cmplx cx2(cdouble(0,0), Nintervals);
    FOR(i, x){
        x2[i] = SQ(x[i]);
        cx2[i] = x2[i];
    }

    // Initialisation de la fonction d'onde :
    vec_cmplx psi(Npoints);

    // initialization time and position to check Probability
    double t = 0;
    unsigned int Nx0 = floor((0 - xL)/(xR-xL)*Npoints); //chosen xR*0.5 since top of potential is at half x domain
  
    // TODO initialize psi
    for (int i(0); i < Npoints; ++i){
        psi[i] = exp(j*k0*x[i])*exp(-SQ(x[i]-x0)/(2*SQ(sigma0)));
    }
   
    // Modifications des valeurs aux bords :
    psi[0] = 0;
    psi[last_diag] = 0;
    
    // Normalisation :
    psi = normalize(psi, dx);

    // Matrices (d: diagonale, a: sous-diagonale, c: sur-diagonale) :
    vec_cmplx dH(Npoints), aH(Nintervals), cH(Nintervals); // matrice Hamiltonienne
    vec_cmplx dA(Npoints), aA(Nintervals),
      cA(Nintervals); // matrice du membre de gauche de l'equation (4.100)
    vec_cmplx dB(Npoints), aB(Nintervals),
      cB(Nintervals); // matrice du membre de droite de l'equation (4.100)

    const cdouble pcoef = -j*hbar/(2*dx);
    vec_cmplx dP(cdouble(0,0), Npoints), aP(cdouble(1,0),Nintervals), cP(cdouble(-1,0),Nintervals);
    dP[0]=-2; dP[last_diag]=2;
    aP[0]=2; cP[last]=-2;
    dP *= pcoef; aP *= pcoef; cP *= pcoef;

    const double p2coef = -SQ(hbar);
    vec_cmplx dP2(cdouble(-2,0), Npoints), aP2(cdouble(1,0), Nintervals), cP2(cdouble(1,0), Nintervals);
    dP2[0] = 0; dP2[last_diag] = 0;
    aP2[0] = 0; cP2[last] = 0;
    dP2 *= p2coef; aP2 *= p2coef; cP2 *= p2coef;

    complex<double> a =
      complex_i * hbar * dt / (4.*m*dx*dx); // Coefficient complexe a de l'equation (4.100)

    // TODO: calculer les éléments des matrices A, B et H.
    // Ces matrices sont stockées sous forme tridiagonale, d:diagonale, c et a: diagonales
    // supérieures et inférieures
    const double H_coef = -SQ(hbar)/(2*m*SQ(dx));
    #define AB_coef(h_) (j*0.5*dt*h_/hbar)
    for (int i(0); i < Npoints; ++i) // Boucle sur les points de maillage
    {
        dH[i] = -2*H_coef + V(x[i], V0, om0, xa, xb, xL, xR, m);
        dA[i] = 1.0 + AB_coef(dH[i]);
        dB[i] = 1.0 - AB_coef(dH[i]);
    }
    for (int i(0); i < Nintervals; ++i) // Boucle sur les intervalles
    {
        aH[i] = H_coef;
        aA[i] = AB_coef(aH[i]);
        aB[i] = -AB_coef(aH[i]);
        cH[i] = H_coef;
        cA[i] = AB_coef(cH[i]);
        cB[i] = -AB_coef(cH[i]);
    }

    // Conditions aux limites: psi nulle aux deux bords
    // TODO: Modifier les matrices A et B pour satisfaire les conditions aux limites
    aA[0] = 0; cA[0] = 0;
    aA[last] = 0; cA[last] = 0;
    aB[0] = 0; cB[0] = 0;
    aB[last] = 0; cB[last] = 0;

    dA[0] = 1; dA[last_diag] = 1;
    dB[0] = 1; dB[last_diag] = 1;

    size_t last_xneg_idx=0;
    while(last_xneg_idx<Npoints-1 && x[last_xneg_idx] < 0 && x[last_xneg_idx+1]<0){
        last_xneg_idx++;
    }

    // Fichiers de sortie :
    string output = configFile.get<string>("output");

    #define NEW_FILE(name, suffix) ofstream name((output + suffix).c_str(), ios::ate | ios::binary)
    NEW_FILE(fichier_potentiel, "_pot.out");
    NEW_FILE(fichier_x, "_x.out");
    NEW_FILE(fichier_psi, "_psi2.out");
    NEW_FILE(fichier_observables, "_obs.out");

    double *V_out = new double[Npoints];
    for (int i(0); i < Npoints; ++i){
        V_out[i] = V(x[i], V0, om0, xa, xb, xL, xR, m);
    }
    double *x_out = new double[Npoints];
    copy(begin(x), end(x), x_out);
    fichier_potentiel.write((char*)V_out, sizeof(double)*Npoints);
    fichier_x.write((char*)x_out, sizeof(double)*Npoints);
    fichier_potentiel.close();
    fichier_x.close();
    delete[] V_out; delete[] x_out;

    /*
    ofstream fichier_psi((output + "_psi2.out").c_str());
    fichier_psi.precision(6);

    ofstream fichier_observables((output + "_obs.out").c_str());
    fichier_observables.precision(15);
    */

        // Ecriture des observables :
    
    // TODO: introduire les arguments des fonctions prob, E, xmoy, x2moy, pmoy et p2moy
    //       en accord avec la façon dont vous les aurez programmés plus haut

    // Boucle temporelle :    
    while (t < tfin) {
        for (int i(0); i < Npoints; ++i){
            writePsi(fichier_psi, psi[i]);
        }

        double prob1, prob2, energy, avg_x, avg_x2, avg_p, avg_p2;
#pragma omp parallel sections
{
#pragma omp section
{       prob1 = prob(psi, dx, 0, last_xneg_idx); }
#pragma omp section
{       prob2 = prob(psi, dx, last_xneg_idx, Npoints-1); }
#pragma omp section
{       energy = E(psi, aH, dH, cH, dx); }
#pragma omp section
{       avg_x = xmoy(psi, cx, zeros, dx); }
#pragma omp section
{       avg_x2 = x2moy(psi, cx2, zeros, dx); }
#pragma omp section
{       avg_p =  pmoy (psi, aP, dP, cP, dx); }
#pragma omp section
{       avg_p2 = p2moy (psi, aP2, dP2, cP2, dx); }
}
        writeObs(fichier_observables, t, prob1, prob2, energy, avg_x, avg_x2, avg_p, avg_p2);
        // Multiplication psi_tmp = B * psi :
        vec_cmplx psi_tmp(cdouble(0,0), Npoints);
        #pragma omp parallel for num_threads(NTHREADS)
        for (size_t i=0; i < Npoints; ++i)
            psi_tmp[i] = dB[i] * psi[i];
        #pragma omp parallel for num_threads(NTHREADS)
        for (size_t i=0; i < Nintervals; ++i) {
            psi_tmp[i] += cB[i] * psi[i + 1];
            psi_tmp[i + 1] += aB[i] * psi[i];
        }

        // Resolution de A * psi = psi_tmp :
        triangular_solve(dA, aA, cA, psi_tmp, psi);
        t += dt;

    } // Fin de la boucle temporelle


    fichier_observables.close();
    fichier_psi.close();

    const auto simulationEnd = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsedSeconds = simulationEnd - simulationStart;
    std::cout << "Simulation finished in " << setprecision(3) << elapsedSeconds.count()
              << " seconds" << std::endl;
}
