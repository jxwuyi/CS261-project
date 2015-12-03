#pragma once
#include "simd.hh"
#include "client.h"

#include <assert.h>
#include <vector>
#include <crypto/paillier.hh>
#include <crypto/gm.hh>
#include <NTL/ZZ.h>
#include <gmpxx.h>
#include <math/util_gmp_rand.h>
#include <limits.h>

#include <fstream>
#include <iostream>
#include <string>

using std::vector;

template<class T>
struct Storage<T> {
  int k; // channel
  int n, m; // size (n x m)
  vector<T> dat;
  int bas1, bas2, sz;
  void init(int _k, int _n, int _m) {
    k = _k; n = _n; m = _m;
    bas1 = n * m;
    bas2 = m;
    sz = k * n * m;
    dat.resize(sz);
  }
  T& at(int x, int y, int z) {
    return dat[x * bas1 + y * bas2 + z];
  }
  int idx(int x,int y,int z){return x*bas1+y*bas2+z;}
  T& at(int x){return dat[x];}
  int size() {return sz;}
};

struct Layer {
  // each filter size
  int n, m;
  // input channel & output channel
  int k_in, k_out;
  
  //////////////////////////////////////
  // NOTE:
  // Assume:
  // dat[i][j] = round(true_dat * base)
  vector<Storage<int> > dat;
  
  // -1: none(prediction); 0: ReLu; type > 0: max pooling dim = type
  int type;
  Layer(){type = -1;}
  
  int bas1, bas2, bas3;
  void load_Layer(FILE *inf, int _k_out, int _k_in, int _n, int _m, int base) {
    k_out = _k_out;
    k_in = _k_in;
    n = _n;
    m = _m;
    dat.clear();
    // TODO: load data from File and store in dat
    //    only keey precision <base>
  }
  void set_type(int t) {
    type = t;
  }
  
  int& at(int a, int b, int c, int d) {
    /*
      a: output channel id
      b: input channel id
      c: filter x
      d: filter y
    */
    return dat[a].at(b,c,d);
  }
  Storage<int>& at(int a){return dat[a];}
};

class DL_Server {
public:
  DL_Server(Paillier* _p, DL_Client_Util* _cl) {
    assert(_p != NULL);
    assert(_cl != NULL);
    p = _p;
    client = _cl;
    
    // default parameters
    shift = 1000; // add <shift> to avoid negative values
    base = 1000; // remain 3 digits
  }
  
  int get_shift() {
    return shift;
  }
  int get_base() {
    return base;
  }
  
  
  void load_model(char * filename);
  
  // classification
  int classify(mpz_class** dat, int n);
  
private:
  
  void clear_temp_params(
    int len, int ndata,
    vector<int>&pos_x,
    vector<int>&pos_y,
    vector<vector<int>>&idx){
      pos_x.clear(); pos_y.clear();
      idx.resize(len);
      for(int j=0;j<len;++j){
        idx[j].resize(ndata);
        for(int i=0;i<ndata;++i)
          idx[j][i] = -1;  
      }
  }
  
  void compute_dot_product(
      vector<mpz_class>& enc_param,
      mpz_class& sum_param,
      vector<vector<int> >&idx, 
      Storage<mpz_class>& next, 
      int ch,
      vector<int>& pos_x,
      vector<int>& pos_y);
  
  vector<Layer> model;
  
  SIMD simd; // default parameters
  
  // approximation parameters
  int shift;
  int base;
  
  Paillier *p; 
  DL_Client_Util *client;
};

