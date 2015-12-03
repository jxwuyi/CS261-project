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

#include<iostream>

using namespace std;
using namespace NTL;


void load_mnist() {
    MNIST mnist;
    mnist.loadData("/home/benzh/Downloads/t10k-images-idx3-ubyte");
}

DL_Client_Util* util;
DL_Server* server;

void run() {
    cout << "Test Paillier Dot Product ...\n" << flush;

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
}

int main() {
    load_mnist();

    run();

    return 0;
}
