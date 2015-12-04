#include "server.h"

#include<algorithm>

using namespace std;

void DL_Server::load_model(const char *filename) {
  model.clear();
  FILE *inf = fopen(filename, "r");
  assert(inf != NULL);
  
  int n_layer;
  fscanf(inf, "%d", &n_layer);
  model.resize(n_layer);
  for(auto&layer: model)
    layer.load(inf, base);
  
  fclose(inf);
}

void DL_Server::compute_dot_product(
  vector<mpz_class>& enc_param,
  mpz_class& sum_param,
  mpz_class& bias,
  vector<vector<int> >&idx, 
  Storage<mpz_class>& next, 
  int ch, // channel
  vector<int>& pos_x,
  vector<int>& pos_y) {
  /////////////////////////
  //  Dot Product
  /////////////////////////
  // get packed values
  vector<mpz_class> packed_values 
      = client->pack_all_from_index(simd, idx);
  // compute dot product
  mpz_class result = p->dot_product(packed_values, enc_param);
  // unpack results
  vector<mpz_class> unpacked_result = client->get_unpack(simd, result);
  
  //////////////////////////
  //  Summation of A
  //////////////////////////
  vector<mpz_class> sumA = client->calc_sum_from_index(idx);
  
  //////////////////////////
  //  Final Results
  //////////////////////////
  // assert(unpacked_result.size() >= pos_x.size())
  vector<mpz_class> target_value 
    = client->normalize(unpacked_result, sumA, sum_param, bias, idx.size(), shift, base);
  for(int i=0;i<pos_x.size();++i) {
    next.at(ch, pos_x[i], pos_y[i]) = target_value[i];
  }
}

int DL_Server::classify(vector<vector<mpz_class> > dat) {
  /*
    Assume: 
      dat[i][j] = encrypt(round(digit[i][j] * base + shift))
  */
  Storage<mpz_class> curr, next;
  curr.init(1, dat.size(), dat[0].size());
  for(int i=0;i<dat.size();++i) {
    for(int j=0;j<dat[0].size();++j)
      curr.at(0,i,j) = dat[i][j];
  }
  
  int ndata = simd.get_ndata();
  
  for(auto&layer: model) {
    //////////////////////////////////////
    // conv layer
    //////////////////////////////////////
    
    /*
      assume:
        1. curr[i][j] = round(true[i][j]*base)+shift
        2. param[i] = layer[i]+shift
            ---> layer[i] = round(true)
     */
    
    // initial size
    next.init(layer.k_out, curr.n - layer.n + 1, curr.m - layer.m + 1);
    
    client->set_cache(curr.dat);
    vector<vector<int> > idx;
    vector<mpz_class> param;
    vector<mpz_class> enc_param;
    vector<int> pos_x, pos_y;
    int len = layer.n*layer.m*layer.k_in;
    clear_temp_params(len, ndata, pos_x, pos_y, idx);
    
    for(int ch = 0; ch < layer.k_out; ++ ch) {
      // set parameters
      mpz_class m_bias(layer.bias[ch]);
      Storage<int> filter = layer.at(ch);
      param.resize(len);
      mpz_class sum_param(0);
      for(int i=0;i<len;++i) {
        param[i] = filter.at(i) + shift;
        sum_param += param[i];
      }
      for(int i=0;i<len;++i) 
        enc_param[i] = p->encrypt(param[i]);
      
      // counters
      int cnt = 0;
      
      for(int x=0;x<next.n; x++)
        for(int y=0;y<next.m; y++) {
          pos_x.push_back(x);
          pos_y.push_back(y);
          
          int ptr = 0;
          for(int k=0;k<filter.k;++k) {
            for(int dx=0;dx<filter.n;dx++)
              for(int dy=0;dy<filter.m;++dy) {
                idx[ptr ++][cnt] = curr.idx(k, x + dx, y + dy);
              }
          }
          // assert(ptr == len)
          if(++ cnt == ndata) {
            compute_dot_product(
              enc_param, sum_param, 
              m_bias,
              idx,
              next, ch,
              pos_x, pos_y);
            clear_temp_params(len, ndata, pos_x, pos_y, idx);
            cnt = 0;
          }
        }
        
        // Remaining
        if(cnt > 0) {
          compute_dot_product(
              enc_param, sum_param,
              m_bias,
              idx,
              next, ch,
              pos_x, pos_y);
          clear_temp_params(len, ndata, pos_x, pos_y, idx);
          cnt = 0;
        }
    }
    
    curr = next;
    
    //////////////////////////////////////
    // Non-Linear Layer
    //////////////////////////////////////
    
    client->set_cache(curr.dat);
    
    if(layer.type < 0) { // prediction/final layer
      return client->get_argmax();
    } else 
    if(layer.type == 0) { // ReLu layer
      mpz_class m_shift(shift);
      client->minus(m_shift); // now client stores all the true values 
      curr.dat = client->get_ReLu(m_shift);
    } else { // Max Pooling Layer
      // Size will change
      int step = layer.type;
      next.init(curr.k, ( curr.n + step - 1 ) / step, ( curr.m + step - 1 ) / step);
      int layer_size = curr.n * curr.m;
      int ptr = 0;
      for(int k=0;k<curr.k;++k) {
        vector<mpz_class> tmp =
          client->get_max(ptr,curr.n,curr.m,step);
        ptr += layer_size;
        int j=0;
        for(int x=0;x<next.n;++x)
          for(int y=0;y<next.m;++y)
            next.at(k,x,y) = tmp[j ++];
      }
      curr = next;
    }
  }
  
  return -1;// should not reach here
}
