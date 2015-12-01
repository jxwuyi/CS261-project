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

    void loadData(string path) {
        uchar** images = read_images(path, num_images_, image_size_);
        // std::cout << "Loaded " << num_images_ << "images "
        // << "with size" << image_size_;
    }

  private:

    int num_images_;
    int image_size_;

    uchar** read_images(string full_path, int& number_of_images, int& image_size) {
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

            }

            file.read((char *) &number_of_images, sizeof(number_of_images));
            number_of_images = reverseInt(number_of_images);

            file.read((char *)&n_rows, sizeof(n_rows));
            n_rows = reverseInt(n_rows);

            file.read((char *)&n_cols, sizeof(n_cols));
            n_cols = reverseInt(n_cols);

            image_size = n_rows * n_cols;

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
};
