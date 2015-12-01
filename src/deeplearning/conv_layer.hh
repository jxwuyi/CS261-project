#pragma once

#include <crypto/paillier.hh>
#include <gmpxx.h>

class ConvLayer {
  public:
    ConvLayer(Paillier p) : paillier_(p) {
    }

    void parseFrom();

    void clientPack() {

    }

    void conv() {
        mpz_class result = p.dot_product(ciphertexts, constants);
    }

  private:
    // Images are encrypted and encoded!
    int image_feature_num_;
    int image_width_;
    int image_height_;
    unsigned char **image_;

    int filter_num_;
    int filter_width_;
    int filter_height_;
    std::vector<mpz_class> filter_;

    Paillier paillier_;
};
