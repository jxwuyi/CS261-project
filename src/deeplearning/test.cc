#include "mnist.hh"
#include "simd.hh"

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


void test_mnist_load() {
    MNIST mnist;
    mnist.loadData(
      "/home/yi/ciphermed/src/deeplearning/t10k-images-idx3-ubyte",
      "/home/yi/ciphermed/src/deeplearning/t10k-labels-idx1-ubyte");
}

void test_paillier_dot_product() {
    cout << "Test Paillier Dot Product ...\n" << flush;

    gmp_randstate_t randstate;
    gmp_randinit_default(randstate);
    gmp_randseed_ui(randstate, time(NULL));

    auto sk = Paillier_priv_fast::keygen(randstate, 1024);
    Paillier_priv_fast pp(sk, randstate);

    auto pk = pp.pubkey();
    mpz_class n = pk[0];
    Paillier p(pk,randstate);

    const int kVectorSize = 100;
    std::vector<mpz_class> constants;
    std::vector<mpz_class> plaintexts;
    std::vector<mpz_class> packed;
    std::vector<mpz_class> ciphertexts;
    long dotsum = 0;

    SIMD simd;

    for (int i = 0; i < kVectorSize; i++) {
        long c = rand();
        int text = i;

        mpz_class pt(std::to_string(text).c_str());
        constants.emplace_back(std::to_string(c).c_str());
        plaintexts.emplace_back(pt);

        mpz_class np(std::to_string(1).c_str());

        vector<mpz_class> v;
        for (int j = 0; j < 16; j++) {
            v.emplace_back(pt);
        }

        mpz_class pack = simd.Pack(v);

        packed.emplace_back(pack);

        dotsum += c * text;
    };

    for (int i = 0; i < kVectorSize; i++) {
        ciphertexts.emplace_back(p.encrypt(packed[i]));
    }

    mpz_class expected(std::to_string(dotsum));

    struct timespec t0,t1;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t0);

    mpz_class result = p.dot_product(ciphertexts, constants);

    mpz_class dr = pp.decrypt(result);
    vector<mpz_class> rs = simd.UnPack(dr);

    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);

    cout << "dotsum = " << dotsum << "" << endl;
    cout << "result = " << rs[0] << "" << endl;
    assert(rs[0] == expected);
    cout << " passed (dotsum = " << dotsum << ")" << endl;

    uint64_t t = (((uint64_t) t1.tv_sec) - ((uint64_t) t0.tv_sec) ) * 1000000000 +
            (t1.tv_nsec - t0.tv_nsec);
    cout << "dot product: "<< ((double) t/1000000) <<" ms per plaintext" << endl;
}

int test_main(int ac, char **av) {
    test_mnist_load();

    test_paillier_dot_product();

    return 0;
}
