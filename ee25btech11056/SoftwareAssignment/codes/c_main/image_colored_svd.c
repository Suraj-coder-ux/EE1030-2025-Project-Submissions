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

// calculation norm of vector
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

  // to check if the file is pgm(grey) or ppm(colored)

  int ks[] = {5, 20, 50, 100};

  int size_k = 4;

  int channels;
  unsigned char *img = stbi_load(filename, &w, &h, &channels, 0);

  if (!img) {
    fprintf(stderr, "Failed to load image: %s\n", filename);
    return 1;
  }

  if (channels == 1) { // greyscale image

    double **A = alloc_matrix(h, w);

    for (int i = 0; i < h; i++)
      for (int j = 0; j < w; j++)
        A[i][j] = img[i * w + j];

    for (int idx = 0; idx < size_k; idx++) {

      int k = ks[idx];

      clock_t start = clock();

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
        // write output image(quality=85 for jpeg)
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

    free_matrix(A, h);

  }

  else if (channels == 3) {

    // allocate matrices for RGB channels
    double **R = alloc_matrix(h, w);
    double **G = alloc_matrix(h, w);
    double **B = alloc_matrix(h, w);

    // extract RGB channels
    for (int i = 0; i < h; i++)
      for (int j = 0; j < w; j++) {
        int idx = (i * w + j) * 3; // pixel is of 3 bytes
        R[i][j] = img[idx + 0];    // Red
        G[i][j] = img[idx + 1];    // Green
        B[i][j] = img[idx + 2];    // Blue
      }

    for (int idx = 0; idx < size_k; idx++) {

      int k = ks[idx];

      clock_t start = clock();

      // apply SVD to each channel
      double **R_k = low_rank(R, h, w, k);
      double **G_k = low_rank(G, h, w, k);
      double **B_k = low_rank(B, h, w, k);

      clock_t end = clock();

      double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

      double err_r = frobenius_norm(R, R_k, h, w);
      double err_g = frobenius_norm(G, G_k, h, w);
      double err_b = frobenius_norm(B, B_k, h, w);

      unsigned char *out = malloc(w * h * 3);

      // merge channels back
      for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++) {
          int pix_idx = (i * w + j) * 3;

          double r_val = R_k[i][j];
          if (r_val < 0)
            r_val = 0;
          if (r_val > 255)
            r_val = 255;
          out[pix_idx + 0] = (unsigned char)(r_val + 0.5);

          double g_val = G_k[i][j];
          if (g_val < 0)
            g_val = 0;
          if (g_val > 255)
            g_val = 255;
          out[pix_idx + 1] = (unsigned char)(g_val + 0.5);

          double b_val = B_k[i][j];
          if (b_val < 0)
            b_val = 0;
          if (b_val > 255)
            b_val = 255;
          out[pix_idx + 2] = (unsigned char)(b_val + 0.5);
        }

      char fname[100];

      if (is_jpg) {
        sprintf(fname, "%s_%d.jpg", base_name, k);
        // write output image
        stbi_write_jpg(fname, w, h, 3, out, 85);
      }

      else {
        sprintf(fname, "%s_%d.png", base_name, k);
        stbi_write_png(fname, w, h, 3, out, w);
      }

      printf("Time for k=%d : %.2f seconds\n", k, time_taken);

      printf("   %3d  | R:%7.2f G:%7.2f B:%7.2f | %s\n", k, err_r, err_g, err_b,
             fname);

      free(out);
      free_matrix(R_k, h);
      free_matrix(G_k, h);
      free_matrix(B_k, h);
    }

    free_matrix(R, h);
    free_matrix(G, h);
    free_matrix(B, h);
  }

  else {
    fprintf(stderr, "Unsupported number of channels : &d\n", channels);
    stbi_image_free(img);
    return 1;
  }

  stbi_image_free(img);

  return 0;
}
