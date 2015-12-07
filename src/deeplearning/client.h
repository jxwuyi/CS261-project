#pragma once

#include <assert.h>
#include <vector>
#include <crypto/paillier.hh>
#include <crypto/gm.hh>
#include <NTL/ZZ.h>
#include <gmpxx.h>
#include <math/util_gmp_rand.h>
#include <limits.h>

#include "simd.hh"

using std::vector;

class DL_Client_Util {
public:
  DL_Client_Util(
    Paillier* _p_enc,
    Paillier_priv_fast * _p_dec){
    p_enc = _p_enc;
    p_dec = _p_dec;
    zero = "0";
    enc_zero = p_enc->encrypt(zero);
  };
  mpz_class pack_from_index(SIMD&simd, vector<int> idx) {
    // if idx(i) < 0 ==> pad zero
    assert(simd.get_ndata() == (int)(idx.size()));
    mpz_class ret;
    vector<mpz_class> input;
    for(int i=0;i<idx.size();++i) {
      if(idx[i] < 0) input.push_back(zero);
      else input.push_back(cache[idx[i]]);
    }
    ret = simd.Pack(input);
    return p_enc->encrypt(ret);
  }
  vector<mpz_class> pack_all_from_index(SIMD&simd, vector<vector<int >> idx) {
    vector<mpz_class> ret;
    for(int i=0;i<idx.size();++i) {
      ret.push_back(pack_from_index(simd, idx[i]));
    }
    return ret;
  }
  vector<mpz_class> calc_sum_from_index(vector<vector<int >> idx) {
    vector<mpz_class> ret;
    int m = idx[0].size();
    ret.resize(m);
    for(int i=0;i<m;++i) ret[i]=zero;
    for(int i=0;i<idx.size();++i)
      for(int j=0;j<m;++j) 
        ret[j] += cache[idx[i][j]];
    for(int i=0;i<m;++i) ret[i] = p_enc->encrypt(ret[i]);
    return ret;
  }
  void set_cache(vector<mpz_class> dat) {
    cache.resize(dat.size());
    for(int i=0;i<dat.size();++i)
      cache[i] = p_dec->decrypt(dat[i]);
  }
  void unpack_cache(SIMD& simd, mpz_class result) {
    mpz_class dr = p_dec->decrypt(result);
    cache = simd.UnPack(dr);
  }
  vector<mpz_class> get_unpack(SIMD& simd, mpz_class result) {
    mpz_class dr = p_dec->decrypt(result);
    vector<mpz_class> ret = simd.UnPack(dr);
    for(int i=0;i<ret.size();++i) {
      ret[i] = p_enc->encrypt(ret[i]);
    }
    return ret;
  }
  
  vector<mpz_class> normalize(
    vector<mpz_class> result, vector<mpz_class> sumA, 
    mpz_class sum_B, 
    mpz_class bias, 
    int _n, int _k, int _base) {
    // K : shift;   n : vector dimension
    // (A + K) * (B + K) = A * B + (sum{A} - n * K) * K + K * B + K * K
    // result & sumA is encrypted
    assert(sumA.size() == result.size());
    mpz_class nk(n * k);
    for(int i=0;i<sumA.size();++i) {
      sumA[i] = p_dec->decrypt(sumA[i]) - nk;
      result[i] = p_dec->decrypt(result[i]);
    }
    mpz_class k(_k), n(_n), base(_base);
    mpz_class nk2 = n * k * k;
    mpz_class k_sB = k * sum_B;
    for(int i=0;i<cache.size();++i) {
      // Compute True Dot Product Result
      result[i] = result[i] - nk2 - k_sB - k *sumA[i];
      // Normalization
      result[i] /= base;
      // Add Bias
      result[i] += bias;
      // Add shift
      result[i] += k;
    }
    
    // encryption
    for(int i=0;i<result.size();++i)
      result[i] = p_enc->encrypt(result[i]);
    return result;
  }
  
  void multiply(mpz_class k) {
    for(int i=0;i<cache.size();++i)
      cache[i] *= k;
  }
  void divide(mpz_class k) {
    for(int i=0;i<cache.size();++i)
      cache[i] /= k;
  }
  void minus(mpz_class k) {
    for(int i=0;i<cache.size();++i)
      cache[i] -= k;
  }
  void add(mpz_class k) {
    for(int i=0;i<cache.size();++i)
      cache[i] += k;
  }
  
  vector<mpz_class> get_max(int start, int n, int m, int step) {
    assert(start + n * m <= cache.size());
    vector<mpz_class> ret;
    for(int i=0;i<n;i+=step)
      for(int j=0;j<m;j+=step) {
        mpz_class val;
        for(int dx=0;dx<step&&i+dx<n;++dx)
          for(int dy=0;dy<step&&j+dy<m;++dy){
            int pos = start + (i+dx)*m + (j+dy);
            if((dx+dy==0)
              || (cmp(val, cache[pos]) < 0))
              val = cache[pos];
          }
        ret.push_back(val);
      }
    
    // Encryption
    for(int i=0;i<ret.size();++i)
      ret[i]=p_enc->encrypt(ret[i]);
    return ret;
  }
  vector<mpz_class> get_ReLu(mpz_class k) {
    vector<mpz_class> ret;
    ret.resize(cache.size());
    mpz_class enc_k = p_enc->encrypt(k);
    for(int i=0;i<cache.size();++i) {
      if(cmp(cache[i], zero) < 0)
        ret[i] = enc_k;
      else
        ret[i] = p_enc->encrypt(cache[i]+k);
    }
    return ret;
  }
  int get_argmax() {
    int pos = 0;
    for(int i=1;i<cache.size();++i) {
      if(cmp(cache[pos], cache[i]) < 0)
        pos = i;
    }
    return pos;
  }
  
  vector<vector<mpz_class> > preprocess(
    unsigned char* image, int n, int m, int shift, int base) {
    // first normalize to -1 ~ 1
    //    [0, 255] --> [0, 1]
    //       x = (x / 255)
    vector<vector<mpz_class> > dat;
    dat.resize(n);
    for(int i=0;i<n;++i) {
      for(int j=0;j<m;++j) {
        int x = (int) image[i * m + j];
        mpz_class m_x((int)(((x / 255.0) * 2 - 1) * base) + shift);
        dat[i].push_back(p_enc->encrypt(m_x));
      }
    }
    return dat;
  }
  
  vector<vector<mpz_class> > preprocess_plain(
    unsigned char* image, int n, int m, int base) {
    // first normalize to -1 ~ 1
    //    [0, 255] --> [0, 1]
    //       x = (x / 255)
    vector<vector<mpz_class> > dat;
    dat.resize(n);
    for(int i=0;i<n;++i) {
      for(int j=0;j<m;++j) {
        int x = (int) image[i * m + j];
        mpz_class m_x((int)((x / 255.0) * base));
        dat[i].push_back(m_x);
      }
    }
    return dat;
  }
  
private:
  mpz_class zero, enc_zero;
  vector<mpz_class> cache;
  
  Paillier *p_enc; // for encryption
  Paillier_priv_fast *p_dec; // for decryption
};

