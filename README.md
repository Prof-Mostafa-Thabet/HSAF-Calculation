HSAF-Calculation

<img src="media/image1.png" id="image1">

Last Update. 2025-9-10 …

What is this repository for?

It is Python-based source code with optimal models for estimating horizontal site amplification factors (HSAF) using solely horizontal-to-vertical spectral ratios of microtremors (MHVSR). The optimal models are based on a newly introduced hybrid machine learning framework following the concept of homogeneous transfer learning. 

Please cite the code as: 

Thabet, M., 2025. Hybrid Machine Learning Models to Calculate Horizontal Site Amplification Factors Using H/V of Microtremors.

Thabet, M., Omar, K., 2025. Hybrid Machine Learning Framework for Horizontal Site Amplification Factors Estimation Using H/V of Microtremors and Earthquakes. Computers & Geoscience (submitted).

How do I get set up?

First, download the source code, example input and output files, PyTorch model, averages of k-means clustering, scaler files for input (x) and output (y) features files.

Then, the models of random forest and gradient boosting are available from the https:// , because these models of random forest and gradient boosting have sizes larger than 2.7 GB, which are larger than the capacity of GitHub. 

Please check the list below (next section) and make sure that you download all these required files. 

Downloaded Documents

| # | # |
|---|---|
|File|Description|
|“HSAF-combined-constrained.py”|Python (Ver. 3.12) source code to run the hybrid machine learning models|
|“control.txt”|Input file list|
|“P5-R.txt”, “P9-R.txt”, “P24-R.txt”, and “P62-R.txt”|Example input files|
|“final_model-best.pth”|PyTorch trained neural network model|
|“gradient_boosting_model-best.joblib”|Gradient boosting trained model|
|“random_forest_model-best.joblib”|Random forest trained model|
|“scaler_X-best.joblib”, “scaler_X_combined-best.joblib”, and “scaler_y-best.joblib”|Scaler files for input (x) and output (y) features files|
|“X_observed5_avg.txt”|Averages of k-means clustering|
|“combined_plot.png”|Example output file|

Usage

Open the “HSAF-combined-constrained.py” and make sure to have all the downloaded files in one directory.

Each input MHVSR file contains eight columns. From the first column through the eighth column, you must have frequency points, first peak frequency, first trough frequency, second peak frequency, second trough peak frequency, average MHVSR, MHVSR-std, MHVSR+std. Note that std is one standard deviation. If the MHVSR contains only one peak frequency, then, make the second peak frequency and second trough peak frequency equal to the first peak frequency and first trough frequency, respectively.   

Please use the example input MHVSR files, then, check that the output HSAFs are the same as in combined_plot.png.

If you want to disable the constrained condition, please disable line 211 in the “HSAF-combined-constrained.py”.

  

Who do I contact?

Mostafa Thabet 

Geology Department, Faculty of Science, Assiut University, Egypt

License

Creative Commons Attribution 4.0 International

The Creative Commons Attribution license allows re-distribution and re-use of a licensed work on the condition that the creator is appropriately credited.
