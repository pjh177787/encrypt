import numpy as np
import math
from PIL import Image
import os
import sys

try:
    import cPickle as pickle
except ImportError:
    import pickle

import random

class MultiLevelScramble:
    def __init__(self, filename=None):
        self.blockSizes = []  # Will be set after loading the first image
        self.keys = []
        self.invKeys = []
        self.revs = []
        self.filename = filename  # Key file to save/load scrambling keys and block sizes

    def genKey(self, blockSize):
        # Generate a key based on the total number of elements in a block
        total_elements = blockSize[0] * blockSize[1] * blockSize[2]
        key = np.arange(total_elements, dtype=np.uint32)
        np.random.shuffle(key)
        return key

    def setKey(self, key, blockSize):
        rev = (key > key.size / 2)
        invKey = np.argsort(key)
        return key, invKey, rev

    def save(self):
        if self.filename:
            with open(self.filename, 'wb') as fout:
                # Save blockSizes and keys
                pickle.dump({'blockSizes': self.blockSizes, 'keys': self.keys}, fout)

    def load(self):
        if self.filename and os.path.exists(self.filename):
            with open(self.filename, 'rb') as fin:
                data = pickle.load(fin)
                self.blockSizes = data['blockSizes']
                self.keys = data['keys']
            self.invKeys = []
            self.revs = []
            for key, blockSize in zip(self.keys, self.blockSizes):
                key, invKey, rev = self.setKey(key, blockSize)
                self.invKeys.append(invKey)
                self.revs.append(rev)
            return True  # Indicate that loading was successful
        return False  # Indicate that loading failed

    def scramble(self, X, key, blockSize, rev):
        # X = self.padding(X, blockSize)
        # print("X (before)=")
        # print(X.shape)
        # print(X)
        # # Save X into a txt file, each element separated by a comma
        # np.savetxt('X.txt', X.flatten(), delimiter=',', fmt='%d')
        # print("key = ")
        # print(key.shape)
        # print(key)
        # np.savetxt('key.txt', key.flatten(), delimiter=',', fmt='%d')
        # print("blockSize = ")
        # print(blockSize)

        X = X.astype(np.uint8)

        s = X.shape
        for i in range(2):  # pad height and width
            t = s[i+1] / blockSize[i]
            d = t - math.floor(t)
            if d > 0:
                paddingSize = int(blockSize[i] * (math.floor(t) + 1) - s[i+1])
                if i == 0:
                    padding = np.tile(X[:, -1:, :, :], (1, paddingSize, 1, 1))
                    X = np.concatenate((X, padding), axis=1)
                elif i == 1:
                    padding = np.tile(X[:, :, -1:, :], (1, 1, paddingSize, 1))
                    X = np.concatenate((X, padding), axis=2)

        # Ensure X is in uint8 format
        X = X.astype(np.uint8)
        # X = self.doScramble(X, key, rev, blockSize)

        s = X.shape
        numBlock = [s[1] // blockSize[0], s[2] // blockSize[1]]
        numCh = blockSize[2]

        # Reshape and transpose to prepare for scrambling
        X = np.reshape(X, (s[0], numBlock[0], blockSize[0],
                           numBlock[1], blockSize[1], numCh))
        X = np.transpose(X, (0, 1, 3, 2, 4, 5))  # (batch, n_blocks_h, n_blocks_w, block_h, block_w, ch)
        X = np.reshape(X, (s[0], numBlock[0], numBlock[1],
                           blockSize[0] * blockSize[1] * numCh))  # Flattened block

        # Apply scrambling key directly on 8-bit pixel values
        if len(key) != X.shape[3]:
            raise ValueError("Key length does not match data dimension.")
        X = X[:, :, :, key]

        # Reshape and transpose back to original format
        X = np.reshape(X, (s[0], numBlock[0], numBlock[1],
                           blockSize[0], blockSize[1], numCh))
        X = np.transpose(X, (0, 1, 3, 2, 4, 5))
        X = np.reshape(X, (s[0],
                           numBlock[0] * blockSize[0],
                           numBlock[1] * blockSize[1], numCh))
        
        # print("X (after) = ")
        # print(X.shape)
        # print(X)
        # np.savetxt('X_after.txt', X.flatten(), delimiter=',', fmt='%d')
        return X

    def scrambleImage(self, data, level=10):
        H, W = data.shape[1], data.shape[2]

        # Generate block sizes based on image dimensions if not already set
        if not self.blockSizes:
            self.generateBlockSizes(H, W, level)

        # Generate keys for the generated block sizes if not already loaded
        if not self.keys:
            for blockSize in self.blockSizes:
                key = self.genKey(blockSize)
                key, invKey, rev = self.setKey(key, blockSize)
                self.keys.append(key)
                self.invKeys.append(invKey)
                self.revs.append(rev)
            # Save keys and block sizes
            self.save()

        # Use the level specified
        if level > len(self.blockSizes):
            raise ValueError(f"Specified level {level} exceeds the number of available block sizes {len(self.blockSizes)}.")

        scrambled_data = data
        for l in range(level):
            scrambled_data = self.scramble(scrambled_data, self.keys[l], self.blockSizes[l], self.revs[l])

        return scrambled_data

    def generateBlockSizes(self, H, W, levels):
        # Generate block sizes based on image size
        max_block_size = min(H, W) // 4  # Max block size is 1/4 of the smallest dimension
        max_block_size = max(max_block_size, 2)  # Ensure at least size 2

        # Generate block sizes as powers of 2
        possible_block_sizes = []
        size = 2
        while size <= max_block_size and len(possible_block_sizes) < levels:
            possible_block_sizes.append(size)
            size *= 2

        # If we still have fewer levels, fill with the max_block_size
        while len(possible_block_sizes) < levels:
            possible_block_sizes.append(max_block_size)

        self.blockSizes = [(bs, bs, 3) for bs in possible_block_sizes]

    def getBlockSizes(self):
        return self.blockSizes

def process_directory(input_dir, output_dir, level, keyFile):
    scrambler = MultiLevelScramble(filename=keyFile)

    # Load the first image to determine dimensions and generate keys if necessary
    first_image_path = None
    print(input_dir)
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            first_image_path = os.path.join(root, file)
            break
        if first_image_path:
            break

    if not first_image_path:
        print("No images found in the input directory.")
        return

    # Load the first image to generate block sizes and keys
    im = Image.open(first_image_path)
    data = np.asarray(im, dtype=np.uint8)
    if len(data.shape) == 2:
        # Convert grayscale images to RGB by stacking
        data = np.stack((data,) * 3, axis=-1)
    data = np.reshape(data, (1,) + data.shape)

    H, W = data.shape[1], data.shape[2]

    if not scrambler.load():
        scrambler.generateBlockSizes(H, W, level)
        scrambler.scrambleImage(data, level=level)  # This will generate and save the keys
        print("Generated and saved keys.")

    # Process all images using the same keys and block sizes
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            input_file_path = os.path.join(root, file)
            # Determine the relative path to maintain directory structure
            rel_path = os.path.relpath(root, input_dir)
            # Create the corresponding output directory if it doesn't exist
            output_dir_full = os.path.join(output_dir, rel_path)
            os.makedirs(output_dir_full, exist_ok=True)
            output_file_path = os.path.join(output_dir_full, file)
            try:
                im = Image.open(input_file_path)
                data = np.asarray(im, dtype=np.uint8)
                if len(data.shape) == 2:
                    # Convert grayscale images to RGB by stacking
                    data = np.stack((data,) * 3, axis=-1)
                data = np.reshape(data, (1,) + data.shape)

                # Verify image dimensions
                H_img, W_img = data.shape[1], data.shape[2]
                if H_img != H or W_img != W:
                    raise ValueError(f"Image dimensions {H_img}x{W_img} do not match the expected dimensions {H}x{W}.")

                # Scramble the image
                scrambled_data = scrambler.scrambleImage(data, level=level)

                # Crop scrambled_data to original image size
                original_height, original_width = data.shape[1], data.shape[2]
                scrambled_data_cropped = scrambled_data[:, :original_height, :original_width, :]

                scrambled_image = Image.fromarray(scrambled_data_cropped[0].astype(np.uint8))
                scrambled_image.save(output_file_path)
                print(f"Scrambled image saved to {output_file_path}")

            except Exception as e:
                print(f"Cannot process file {input_file_path}: {e}")

if __name__ == '__main__':
    # Specify the scrambling level (e.g., level = 10)
    level = 1  # User-specified level between 1 and 10

    # Input and output directories
    input_dir = '/Users/aperture/Git/encrypt/data/test'  # Replace with your input directory path
    output_dir = '/Users/aperture/Git/encrypt/py_test/scrambled_' + str(level) + '_pixel'  # Replace with your output directory path
    keyFile = 'lv' + str(level) + '_' + 'keys_with_block_sizes.pkl'  # Key file to save/load scrambling keys and block sizes

    process_directory(input_dir, output_dir, level, keyFile)

    # Optionally, print the block sizes used
    print("Block sizes used for scrambling:")
    scrambler = MultiLevelScramble(filename=keyFile)
    for idx, blockSize in enumerate(scrambler.getBlockSizes(), start=1):
        print(f"Level {idx}: Block size {blockSize}")
