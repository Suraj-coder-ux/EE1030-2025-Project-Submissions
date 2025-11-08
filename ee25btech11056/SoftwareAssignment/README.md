# Image Compression Using Truncated SVD

This project implements image compression using Singular Value Decomposition (SVD) with power iteration algorithm. The implementation decomposes grayscale and color images into their singular value components and reconstructs them using only the top k singular values, achieving significant compression while maintaining visual quality. The power iteration method with deflation is used to compute dominant singular vectors iteratively. The project demonstrates the mathematical foundation of SVD-based compression and includes comparison with NumPy's optimized SVD implementation, showing the trade-offs between compression ratio and image fidelity.

## Features
- Grayscale and RGB image compression
- Power iteration algorithm for SVD computation
- Support for multiple image formats (PNG, JPG)
- Runtime comparison with NumPy's SVD
- Frobenius norm error calculation

Enter image filename as input.The code outputs compressed images for k = 5, 20, 50, 100.
