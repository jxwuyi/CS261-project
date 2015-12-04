#include "mnist.hh"
#include "client.h"
#include "server.h"

#include <assert.h>
#include <vector>
#include <crypto/paillier.hh>
#include <crypto/gm.hh>
#include <NTL/ZZ.h>
#include <gmpxx.h>
#include <math/util_gmp_rand.h>
#include <limits.h>

#include <ctime>
#include <chrono>

#include<iostream>

using namespace std;
//using namespace NTL;

const int ReportStep = 1000;


DL_Client_Util* util;
DL_Server* server;
MNIST mnist;

void load_mnist() {
    mnist.loadData(
      "/home/yi/ciphermed/src/deeplearning/t10k-images-idx3-ubyte",
      "/home/yi/ciphermed/src/deeplearning/t10k-labels-idx1-ubyte");
}

void run() {
    gmp_randstate_t randstate;
    gmp_randinit_default(randstate);
    gmp_randseed_ui(randstate, time(NULL));

    auto sk = Paillier_priv_fast::keygen(randstate, 1024);
    Paillier_priv_fast pp(sk, randstate);

    auto pk = pp.pubkey();
    Paillier p(pk,randstate);
    
    // Launch Client Util
    util = new DL_Client_Util(&p, &pp);
    
    // Launch Server
    server = new DL_Server(&p, util);
    server->load_model(
      "/home/yi/ciphermed/src/deeplearning/model_weights.txt");
    
    cout << "Server and Client successfully launched!" << endl;
    
    int total = mnist.size();
    int n_row = mnist.row();
    int n_col = mnist.col();
    int base = server->get_base();
    int correct = 0;
    int wrong = 0;
    
    std::chrono::time_point<std::chrono::system_clock> __start_time 
      = std::chrono::system_clock::now();
    
    for(int i=1;i<=total;++i){
      int ans = mnist.label(i);
      int rec = server->classify(
        util->preprocess(mnist.image(i), n_row, n_col, base));
      if (ans == rec) ++ correct;
      else ++ wrong;
      
      if (i % ReportStep == 0 || i == total) {
        // Report Result
        cout << "Image "<<i << " / total ("<< ((int)(i*100.0/total)) <<"\%)" << endl;
        printf( "  >> Correct = %d (%.2lf\%)\n", correct, (double)(100.0*correct/i));
        printf( "  >> Wrong = %d (%.2lf\%)\n", wrong, (double)(100.0*wrong/i));
        
        std::chrono::duration<double> __elapsed_seconds = 
            std::chrono::system_clock::now()-__start_time;
        printf( "   --> Time Elasped = %lfs\n", (double)__elapsed_seconds.count());
      }
      
    }
    
    
    delete util;
    delete server;
}

int main() {
    load_mnist();
    cout << "MNIST successfully loaded!" << endl;

    run();

    return 0;
}
