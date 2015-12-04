#pragma once

#include <fstream>
#include <iostream>
#include <string>

using std::string;
typedef unsigned char uchar;

class MNIST {
  public:
    MNIST() {
    }

    void loadData(string image_path, string label_path) {
        images = read_images(image_path, num_images_, image_size_, image_row_, image_col_);
        // std::cout << "Loaded " << num_images_ << "images "
        // << "with size" << image_size_;
        labels = read_labels(label_path);
    }
    
    int size() {return num_images;}
    int n_row() {return image_row_;}
    int n_col() {return image_col_;}
    uchar* image(int k) {return images[k];}
    uchar label(int k){return labels[k];}

  private:

    int num_images_;
    int image_size_;
    int image_row_;
    int image_col_;
    uchar** images;
    uchar* labels;

    uchar** read_images(string full_path, int& number_of_images, 
        int& image_size, int& image_row, int& image_col) {
        auto reverseInt = [](int i) {
            unsigned char c1, c2, c3, c4;
            c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
            return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
        };

        std::ifstream file(full_path);

        if (file.is_open()) {
            int magic_number = 0, n_rows = 0, n_cols = 0;

            file.read((char *)&magic_number, sizeof(magic_number));
            magic_number = reverseInt(magic_number);

            if (magic_number != 2051) {
              std::cerr << "Magic number is wrong in <"<<full_path<<std::endl;
            }

            file.read((char *) &number_of_images, sizeof(number_of_images));
            number_of_images = reverseInt(number_of_images);

            file.read((char *)&n_rows, sizeof(n_rows));
            n_rows = reverseInt(n_rows);

            file.read((char *)&n_cols, sizeof(n_cols));
            n_cols = reverseInt(n_cols);

            image_size = n_rows * n_cols;
            image_row = n_rows;
            image_col = n_cols;

            uchar** _dataset = new uchar*[number_of_images];
            for(int i = 0; i < number_of_images; i++) {
                _dataset[i] = new uchar[image_size];
                file.read((char *)_dataset[i], image_size);
            }
            return _dataset;
        } else {
            std::terminate();
        }
    }
    
    uchar* read_labels(string full_path) {
        auto reverseInt = [](int i) {
            unsigned char c1, c2, c3, c4;
            c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
            return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
        };

        std::ifstream file(full_path);

        if (file.is_open()) {
            int magic_number = 0, n_images = 0;

            file.read((char *)&magic_number, sizeof(magic_number));
            magic_number = reverseInt(magic_number);

            if (magic_number != 2049) {
              std::cerr << "Magic number is wrong in <"<<full_path<<std::endl;
            }

            file.read((char *) &n_images, sizeof(n_images));
            n_images = reverseInt(n_images);

            uchar* _labels = new uchar[n_images];
            file.read((char*) _labels, n_images);
            return _labels;
        } else {
            std::terminate();
        }
    }
};
