#define STB_IMAGE_IMPLEMENTATION
#include "../c_libs/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../c_libs/stb_image_write.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

double **alloc_matrix(int m, int n) {

  double **A = malloc(m * sizeof(double *));

  for (int i = 0; i < m; i++)
    A[i] = calloc(
        n,
        sizeof(double)); // calloc allocates memory and initializes values to 0

  return A;
}

void free_matrix(double **A, int m) {

  for (int i = 0; i < m; i++)
    free(A[i]);

  free(A);
}

// Matrix-vector multiplication
void matvec(double **A, double *x, double *y, int m, int n) {

  for (int i = 0; i < m; i++) {
    y[i] = 0;

    for (int j = 0; j < n; j++)
      y[i] += A[i][j] * x[j];
  }
}

// Matrix_transpose-vector multiplication
void mat_t_vec(double **A, double *x, double *y, int m, int n) {

  for (int j = 0; j < n; j++) {
    y[j] = 0;

    for (int i = 0; i < m; i++)
      y[j] += A[i][j] * x[i];
  }
}

// calculation of norm of vector
double norm(double *x, int n) {

  double s = 0;

  for (int i = 0; i < n; i++)
    s += x[i] * x[i];

  return sqrt(s);
}

// truncated svd - low_rank-k approximation
double **low_rank(double **A, int m, int n, int k) {

  double **Ak = alloc_matrix(m, n); // final compressed image
  double **Atemp = alloc_matrix(m, n);

  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      Atemp[i][j] = A[i][j]; // copy matrix A to Atemp

  double *v = malloc(n * sizeof(double));
  double *u = malloc(m * sizeof(double));
  double *temp = malloc(m * sizeof(double));

  for (int t = 0; t < k; t++) {

    for (int j = 0; j < n; j++)
      v[j] = (double)rand() / RAND_MAX; // to type cast int to double , a number
                                        // between 0 and 1 is stored in v[j]

    // normalize v
    double nv = norm(v, n);

    for (int j = 0; j < n; j++)
      v[j] /= nv;

    // power iteration to get dominant singular vectors
    for (int iter = 0; iter < 50; iter++) { // perform iterations 50 times

      // temp = A*v
      matvec(Atemp, v, temp, m, n);

      // v = A^T * temp (this means A^T*A*v)
      mat_t_vec(Atemp, temp, v, m, n);

      nv = norm(v, n);

      if (nv < 1e-10) // if norm is too small
        break;

      for (int j = 0; j < n; j++)
        v[j] /= nv;
    }

    // compute u = A*v
    matvec(Atemp, v, u, m, n);

    // sigma = \\u\\ = \\Av||
    double sigma = norm(u, m);

    if (sigma < 1e-10)
      break;

    // normalize u
    for (int i = 0; i < m; i++)
      u[i] /= sigma;

    // add sigma*u*v^T
    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++)
        Ak[i][j] += sigma * u[i] * v[j];

    // deflating : subtract sigma*u*v^T from Atemp
    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++)
        Atemp[i][j] -= sigma * u[i] * v[j];
  }

  free_matrix(Atemp, m);

  free(temp);
  free(v);
  free(u);

  return Ak;
}

// calculate frobenius norm
double frobenius_norm(double **A, double **B, int m, int n) {

  double b = 0;

  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) {
      double d = A[i][j] - B[i][j];
      b += d * d;
    }

  return sqrt(b);
}

const char *get_extension(const char *filename) {

  const char *dot = strrchr(filename, '.');

  if (!dot || dot == filename)
    return "";

  return dot + 1;
}

int main() {

  int w, h;

  char filename[100];
  printf("Enter the input filename: ");
  scanf("%s", filename);

  // get the file extension
  const char *ext = get_extension(filename);
  int is_png = (strcasecmp(ext, "png") == 0);
  int is_jpg = (strcasecmp(ext, "jpg") == 0 || strcasecmp(ext, "jpeg") == 0);

  if (!is_png && !is_jpg) {
    fprintf(stderr, "Unsupported format\n");
    return 1;
  }

  // remove the extension from the filename
  char base_name[100];
  strcpy(base_name, filename);
  char *dot = strrchr(base_name,
                      '.'); // to search(using strrchr) for last occurrence of
                            // (.) operator and returns pointer to this position
  if (dot)
    *dot = '\0'; // remove extension , if dot is found make it null character to
                 // mark end of base_name

  // read image
  int read;
  unsigned char *img = stbi_load(filename, &w, &h, &read, 1);

  if (!img) { // if img is null
    fprintf(stderr, "Failed to load image: %s\n", filename);
    return 1;
  }

  double **A = alloc_matrix(h, w);

  for (int i = 0; i < h; i++)
    for (int j = 0; j < w; j++)
      A[i][j] = img[i * w + j];

  int ks[] = {5, 20, 50, 100};

  int size_k = 4;

  for (int idx = 0; idx < size_k; idx++) {

    int k = ks[idx];

    clock_t start = clock(); // to begin the clock

    double **A_k = low_rank(A, h, w, k);

    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    double err = frobenius_norm(A, A_k, h, w);

    unsigned char *out = malloc(w * h);

    for (int i = 0; i < h; i++)
      for (int j = 0; j < w; j++) {
        double v = A_k[i][j];

        if (v < 0)
          v = 0;

        if (v > 255)
          v = 255;

        out[i * w + j] = (unsigned char)(v + 0.5);
      }

    char fname[150]; // for output filename

    if (is_jpg) {
      sprintf(fname, "%s_%d.jpg", base_name, k);
      // to write output image (quality=85 for jpeg)
      stbi_write_jpg(fname, w, h, 1, out, 85);
    }

    else {
      sprintf(fname, "%s_%d.png", base_name, k);
      stbi_write_png(fname, w, h, 1, out, w); // stride_bytes=w
    }

    printf("Time for k=%d : %.2f seconds\n", k, time_taken);
    printf("   %3d  |  %10.4f  |  %s\n", k, err, fname);

    free(out);
    free_matrix(A_k, h);
  }

  stbi_image_free(img);
  free_matrix(A, h);
  return 0;
}
