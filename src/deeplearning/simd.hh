#pragma once

#include <gmpxx.h>
#include <assert.h>
#include <algorithm>

class SIMD {
  public:
    SIMD(int nbits = 64, int ndata = 16) :
            nbits_(nbits), ndata_(ndata) {
        // For now, we only support 64 bits packing 16 data, leading
        // to 1024 bit data.
        assert(nbits == 64);
        assert(ndata == 16);
        
        nbits_ = nbits;
        ndata_ = ndata;

        mpz_t m1;
        mpz_init_set_ui(m1, 2);
        mpz_init(multiplier_);
        mpz_pow_ui(multiplier_, m1, nbits);

        mpz_init(packed_);
    }

    int get_ndata() {return ndata_;}

    mpz_class Pack(std::vector<mpz_class> inputs) {
        mpz_init_set_ui(packed_, 1);
        size_t size = inputs.size();
        for (size_t i = 0; i < size - 1; i++) {
            mpz_mul(packed_, inputs[i].get_mpz_t(), multiplier_);
        }
        mpz_add(packed_, inputs[size - 1].get_mpz_t(), multiplier_);
        return mpz_class(packed_);
    }

    std::vector<mpz_class> UnPack(mpz_class input) {
        std::vector<mpz_class> result;

        mpz_t new_input;
        mpz_init(new_input);
        mpz_t q;
        mpz_init(q);
        mpz_t r;
        mpz_init(r);

        mpz_set(new_input, input.get_mpz_t());
        for (int i = 0; i < ndata_; i++) {
            mpz_tdiv_qr(q, r, new_input, multiplier_);
            mpz_set(new_input, q);
            result.emplace_back(mpz_class(r));
        }
        std::reverse(result.begin(), result.end());
        
        return result;
    }
  private:
    int nbits_;
    int ndata_;
    mpz_t multiplier_;
    mpz_t packed_;
};
