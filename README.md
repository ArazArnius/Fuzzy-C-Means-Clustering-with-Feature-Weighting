# Fuzzy C-Means Clustering with Feature Weighting

This repository contains a Python implementation of the Fuzzy C-Means (FCM) clustering algorithm with feature weighting. FCM is a popular clustering algorithm used in various domains for partitioning datasets into clusters based on similarity. This implementation extends FCM to incorporate feature weighting, allowing for the consideration of diverse feature importance in the clustering process. This was a task assigned to me in May 2023 during my university studies at Azarbaijan Shahid Madani University in Tabriz, Iran. The objective was to gain proficiency in the NumPy library within the Python programming language. While the course focused on fundamental programming concepts rather than machine learning, this project involved implementing the clustering algorithm with feature weighting as practical exercise. It's worth noting that as a learning endeavor, there may be areas for improvement or refinement in the algorithm. Your understanding and feedback are appreciated as part of the ongoing learning process...

## Usage

To use the code, follow these steps:

1. Ensure you have Python installed on your system.
2. Download or clone this repository to your local machine.
3. Run the Python script `fuzzy_cmeans.py`.
4. The script reads a dataset from a CSV file named `iris.csv`, which contains the data to be clustered. You may need to adjust the file path or provide your own dataset.
5. The algorithm parameters such as the number of clusters (`K`), maximum iterations (`t_max`), fuzzy degree (`alpha`), and exponent of attribute weight (`q`) are configurable and can be adjusted in the script.
6. The script will perform FCM clustering with feature weighting and output the final cluster assignments (`U`), cluster centroids (`C`), and feature weights (`W`).
7. The cluster assignments (`U`) and cluster centroids (`C`) will be saved in CSV files named `U.csv` and `C.csv`, respectively.

## Requirements

- Python 3.x
- NumPy

## References

- V. N. Vapnik and V. Vapnik, Statistical learning theory. Wiley New York, 1998.
- J. Han, J. Pei, and M. Kamber, Data mining: concepts and techniques. Elsevier, 2011.
- J. MacQueen, "Some methods for classification and analysis of multivariate observations," in Proceedings of the fifth Berkeley symposium on mathematical statistics and probability, 1967, vol. 1, no. 14: Oakland, CA, USA., pp. 281-297.
- J. C. Bezdek, Pattern recognition with fuzzy objective function algorithms. Springer Science & Business Media, 2013.
- [P. Huang and D. Zhang, "Locality sensitive C-means clustering algorithms," Neurocomputing, vol. 73, no. 16, pp. 2935-2943, 2010/10/01/ 2010.](https://doi.org/10.1016/j.neucom.2010.07.015)
- T. Kanungo, D. M. Mount, N. S. Netanyahu, C. D. Piatko, R. Silverman, and A. Y. Wu, "An efficient k-means clustering algorithm: Analysis and implementation," IEEE transactions on pattern analysis and machine intelligence, vol. 24, no. 7, pp. 881-892, 2002.
- Y. Liu, F. Tian, Z. Hu, and C. DeLisi, "Evaluation and integration of cancer gene classifiers: identification and ranking of plausible drivers," Scientific reports, vol. 5, 2015.
- L. Zhu, F.-L. Chung, and S. Wang, "Generalized fuzzy c-means clustering algorithm with improved fuzzy partitions," IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), vol. 39, no. 3, pp. 578-591, 2009.
- R. Xu and D. Wunsch, "Survey of clustering algorithms," IEEE Transactions on neural networks, vol. 16, no. 3, pp. 645-678, 2005.
- [L. Parsons, E. Haque, and H. Liu, "Subspace clustering for high dimensional data: a review," SIGKDD Explor. Newsl., vol. 6, no. 1, pp. 90-105, 2004.](https://doi.org/10.1145/1007730.1007731)
- [D. S. Modha and W. S. Spangler, "Feature Weighting in k-Means Clustering," Machine Learning, journal article vol. 52, no. 3, pp. 217-237, September 01 2003.](https://doi.org/10.1023/a:1024016609528)
- J. Z. Huang, M. K. Ng, H. Rong, and Z. Li, "Automated variable weighting in k-means type clustering," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 27, no. 5, pp. 657-668, 2005.
- [X.-b. Zhi, J.-l. Fan, and F. Zhao, "Robust local feature weighting hard c-means clustering algorithm," Neurocomputing, vol. 134, pp. 20-29, 2014/06/25/ 2014.](https://doi.org/10.1016/j.neucom.2012.12.074)

