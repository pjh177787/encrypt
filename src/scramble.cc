#include <cstdint.h>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <cstring>   // For memcpy
#include <iostream>  // For I/O operations
#include <fstream>   // For file operations

void scramble(
    uint8_t X[480][640][3],
    const uint8_t key[12],
    const uint8_t block_size[3],
    bool rev
) {
    // Get the dimensions of X
    const int height = 480;
    const int width = 640;
    const int channels = 3;

    // Calculate padding for height and width
    int padded_height = height;
    int padded_width = width;

    int pad_height = (block_size[0] - (height % block_size[0])) % block_size[0];
    int pad_width = (block_size[1] - (width % block_size[1])) % block_size[1];

    padded_height += pad_height;
    padded_width += pad_width;

    // Create a new array to hold the padded data
    const int max_padded_height = 480 + block_size[0];
    const int max_padded_width = 640 + block_size[1];
    uint8_t X_padded[max_padded_height][max_padded_width][3];

    // Initialize X_padded with zeros
    std::memset(X_padded, 0, sizeof(X_padded));

    // Copy original data into X_padded
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            for (int c = 0; c < channels; ++c) {
                X_padded[h][w][c] = X[h][w][c];
            }
        }
    }

    // Pad the bottom rows if necessary
    for (int h = height; h < padded_height; ++h) {
        for (int w = 0; w < width; ++w) {
            for (int c = 0; c < channels; ++c) {
                X_padded[h][w][c] = X[height - 1][w][c];  // Replicate last row
            }
        }
    }

    // Pad the rightmost columns if necessary
    for (int h = 0; h < padded_height; ++h) {
        for (int w = width; w < padded_width; ++w) {
            for (int c = 0; c < channels; ++c) {
                X_padded[h][w][c] = X_padded[h][width - 1][c];  // Replicate last column
            }
        }
    }

    // Calculate the number of blocks in height and width
    int numBlockH = padded_height / block_size[0];
    int numBlockW = padded_width / block_size[1];
    int block_area = block_size[0] * block_size[1] * channels;

    // Check if the key length matches the block size
    if (12 != block_area) {
        throw std::invalid_argument("Key length does not match data dimension.");
    }

    // Scramble each block
    for (int nb_h = 0; nb_h < numBlockH; ++nb_h) {
        for (int nb_w = 0; nb_w < numBlockW; ++nb_w) {
            // Extract the block
            uint8_t block[12];  // As per your key size

            int idx = 0;
            for (int bh = 0; bh < block_size[0]; ++bh) {
                for (int bw = 0; bw < block_size[1]; ++bw) {
                    for (int c = 0; c < channels; ++c) {
                        int h_idx = nb_h * block_size[0] + bh;
                        int w_idx = nb_w * block_size[1] + bw;
                        block[idx++] = X_padded[h_idx][w_idx][c];
                    }
                }
            }

            // Apply the scrambling key
            uint8_t scrambled_block[12];
            for (int i = 0; i < 12; ++i) {
                scrambled_block[i] = block[key[i]];
            }

            // Place the scrambled block back into X_padded
            idx = 0;
            for (int bh = 0; bh < block_size[0]; ++bh) {
                for (int bw = 0; bw < block_size[1]; ++bw) {
                    for (int c = 0; c < channels; ++c) {
                        int h_idx = nb_h * block_size[0] + bh;
                        int w_idx = nb_w * block_size[1] + bw;
                        X_padded[h_idx][w_idx][c] = scrambled_block[idx++];
                    }
                }
            }
        }
    }

    // Copy the scrambled data back to X (excluding padding)
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            for (int c = 0; c < channels; ++c) {
                X[h][w][c] = X_padded[h][w][c];
            }
        }
    }
}

int main() {
    // Define the dimensions
    const int height = 480;
    const int width = 640;
    const int channels = 3;
    const uint8_t block_size[3] = {2, 2, 3};

    // Declare arrays
    uint8_t X[480][640][3];
    uint8_t X_after[480][640][3];
    uint8_t key[12];

    // Read X from X.txt
    std::ifstream x_file("X.txt");
    if (!x_file) {
        std::cerr << "Error: Cannot open X.txt" << std::endl;
        return 1;
    }
    for (int i = 0; i < height; ++i) {
        for (int y = 0; y < width; ++y) {
            for (int k = 0; k < channels; ++k) {
                int value;
                if (!(x_file >> value)) {
                    std::cerr << "Error: Not enough data in X.txt" << std::endl;
                    return 1;
                }
                X[i][y][k] = static_cast<uint8_t>(value);
            }
        }
    }
    x_file.close();

    // Read key from key.txt
    std::ifstream key_file("key.txt");
    if (!key_file) {
        std::cerr << "Error: Cannot open key.txt" << std::endl;
        return 1;
    }
    for (int i = 0; i < 12; ++i) {
        int value;
        if (!(key_file >> value)) {
            std::cerr << "Error: Not enough data in key.txt" << std::endl;
            return 1;
        }
        key[i] = static_cast<uint8_t>(value);
    }
    key_file.close();

    // Read X_after from X_after.txt
    std::ifstream x_after_file("X_after.txt");
    if (!x_after_file) {
        std::cerr << "Error: Cannot open X_after.txt" << std::endl;
        return 1;
    }
    for (int i = 0; i < height; ++i) {
        for (int y = 0; y < width; ++y) {
            for (int k = 0; k < channels; ++k) {
                int value;
                if (!(x_after_file >> value)) {
                    std::cerr << "Error: Not enough data in X_after.txt" << std::endl;
                    return 1;
                }
                X_after[i][y][k] = static_cast<uint8_t>(value);
            }
        }
    }
    x_after_file.close();

    // Call the scramble function
    try {
        scramble(X, key, block_size, false);
    } catch (const std::exception& e) {
        std::cerr << "Error during scrambling: " << e.what() << std::endl;
        return 1;
    }

    // Compare X and X_after
    bool identical = true;
    for (int i = 0; i < height; ++i) {
        for (int y = 0; y < width; ++y) {
            for (int k = 0; k < channels; ++k) {
                if (X[i][y][k] != X_after[i][y][k]) {
                    identical = false;
                    std::cout << "Difference at position [" << i << "][" << y << "][" << k << "]: "
                              << "X = " << static_cast<int>(X[i][y][k]) << ", "
                              << "X_after = " << static_cast<int>(X_after[i][y][k]) << std::endl;
                    // You can uncomment the next line to stop after the first difference
                    // return 1;
                }
            }
        }
    }

    if (identical) {
        std::cout << "The scrambled X matches X_after.txt. The scramble function works correctly." << std::endl;
    } else {
        std::cout << "The scrambled X does not match X_after.txt." << std::endl;
    }

    return 0;
}
