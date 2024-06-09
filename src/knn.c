#include "knn.h"

/** 
 * Return the euclidean distance between the image pixels (as vectors).
 */
double distance(Image *a, Image *b) {
  int total_pixels = a->sx * a->sy;
  double dist = 0;
  for(int i = 0; i<total_pixels; i++) {
    dist += pow(a->data[i] - b->data[i], 2);
  }
  dist = sqrt(dist);
  return dist; 
}
/** 
 * Returns the index of the largest distance.
 */
int get_largest(double* k_dist, int K) {
  double largest_dist = 0;
  int largest_index = 0;
  
  for(int i = 0; i<K; i++) {
    if(k_dist[i] > largest_dist) {
      largest_dist = k_dist[i];
      largest_index = i;
    }
  }
  return largest_index;
}

/**
 * Returns the mode of k_label.
 */
int mode(int* k_label, int K) {
  int max_count = 0;
  int max_label = 0;
  int counter = 0;

  for(int i = 0; i<10; i++) {
    for(int j = 0; j<K; j++) {
      if(i == k_label[j]) {
        counter++;
      }
    }
    if (counter > max_count) {
      max_count = counter;
      max_label = i;
    }
    counter = 0;
  }
  return max_label;
}

/**
 * Given the input training dataset, an image to classify and K,
 *   (1) Find the K most similar images to `input` in the dataset
 *   (2) Return the most frequent label of these K images
 * 
 * Note: If there's multiple images with the same smallest values, pick the
 *      ones that come first. For automarking we will make sure the K smallest
 *      ones are unique, so it doesn't really matter.
 */ 
int knn_predict(Dataset *data, Image *input, int K) {
  // Find min length
  int min_length;
  if (K < data->num_items) {
    min_length = K;
  } else {
    min_length = data->num_items;
  }

  // 1. Create an array of K to save labels
  double k_dist[min_length]; 
  int k_label[min_length];
  
  // 2. Find dist of input and every item in images array
  for(int i = 0; i<data->num_items; i++) {
    double dist = distance(&data->images[i], input);
    if(i < min_length){
      k_dist[i] = dist;
      k_label[i] = data->labels[i];
    } else { 
      int index_largest = get_largest(k_dist, min_length);
      if(dist < k_dist[index_largest]) {
        k_dist[index_largest] = dist;
        k_label[index_largest] = data->labels[i];
      }
    }
  }
  // 4. Find and return the mod - note if no mode or multiple return the smallest value
  return mode(k_label, min_length);
}

/**
 * This function takes in the name of the binary file containing the data and
 * loads it into memory. The binary file format consists of the following:
 *
 *     -   4 bytes : `N`: Number of images / labels in the file
 *     -   1 byte  : Image 1 label
 *     - 784 bytes : Image 1 data (28x28)
 *          ...
 *     -   1 byte  : Image N label
 *     - 784 bytes : Image N data (28x28)
 *
 * You can simply set the `sx` and `sy` values for all the images to 28, we
 * will not test this with different image sizes.
 */
Dataset *load_dataset(const char *filename) {
  int image_dimension = 28;
  int total_pixels = 28 * 28;
  Dataset* data_set = calloc(1, sizeof(Dataset));

  // Get total number of images
  FILE* list_file = fopen(filename, "rb");
  if(list_file == NULL) {
    return NULL;
  }
  if(fread(&(data_set->num_items), sizeof(int), 1, list_file) == 0) {
    return NULL;
  } 
  data_set->images = (Image*)calloc(data_set->num_items, sizeof(Image));
  data_set->labels = (unsigned char*)calloc(data_set->num_items, sizeof(unsigned char));

  // Load pixel data for every image
  for(int i = 0; i<data_set->num_items; i++) {
    if(fread(&(data_set->labels[i]), sizeof(unsigned char), 1, list_file) == 0) {
      return NULL;
    } // read image label
    data_set->images[i].sx = image_dimension;
    data_set->images[i].sy = image_dimension;
    data_set->images[i].data = (unsigned char*)calloc(total_pixels, sizeof(unsigned char));
    for(int j = 0; j<total_pixels; j++) {
      if(fread(&(data_set->images[i].data[j]), sizeof(unsigned char), 1, list_file) == 0) {
        return NULL;
      }
    }
  }
  fclose(list_file);
  return data_set;
}

/**
 * Free all the allocated memory for the dataset
 */
void free_dataset(Dataset *data) {
  for(int i = 0; i < data->num_items; i++) {
    free(data->images[i].data);
  }
  free(data->images);
  free(data->labels);
  free(data);
  return;
}

/**
 * This function should be called by each child process, and is where the 
 * kNN predictions happen. Along with the training and testing datasets, the
 * function also takes in 
 *    (1) File descriptor for a pipe with input coming from the parent: p_in
 *    (2) File descriptor for a pipe with output going to the parent:  p_out
 * 
 * Once this function is called, the child should do the following:
 *    - Read an integer `start_idx` from the parent (through p_in)
 *    - Read an integer `N` from the parent (through p_in)
 *    - Call `knn_predict()` on testing images `start_idx` to `start_idx+N-1`
 *    - Write an integer representing the number of correct predictions to
 *        the parent (through p_out)
 */
void child_handler(Dataset *training, Dataset *testing, int K, 
                   int p_in, int p_out) {
  int start_idx, N;
  if(read(p_in, &start_idx, sizeof(int)) == -1) {
    perror("read pipe");
  }
  if (read(p_in, &N, sizeof(int)) == -1) {
    perror("read pipe");
  }
  
  int total_correct = 0;
  for(int i = start_idx; i < (start_idx + N); i++) {
    int label_predict = knn_predict(training, &testing->images[i], K);
    if(label_predict == testing->labels[i]) {
      total_correct++;
    }
  }

  if (write(p_out, &total_correct, sizeof(int)) == -1) {
    perror("write to pipe");
  }
  return;
}