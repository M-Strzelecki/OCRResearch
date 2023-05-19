# OCR Research

## Research on implementing OCR to extract data from nutritional labels

This research is a starting point for my implementation of OCR to gather data from images of food labels. In this project we will examine the data from the food labels and extract it from the images. Project contains the following:
[Project Brief Description](/readmefiles/Dietary%20AI%20Assistant.pdf)
* Basic Pipeline to extract data from the food labels
* Testing different types of preprocessing such as:
    * Gamma
    * Thresholding
    * Resizing
* Looking at the image histogram (checking is bimodial or not)
* Comparison of extracted text from the food labels to the hard-coded text
    * Calculating character counts and avarages
    * Calculating SMAPE (Symmetric mean absolute percentage error) and avarages
* Testing automation to choose best parameters for gamma and thresholding 

## Requirements
Python 3 or newer with the following packages installed
| Package          | Version   |
| ---------------- | --------- |
| matplotlib       | 3.7.1     |
| numpy            | 1.24.2    |
| pandas           | 2.0.0     |
| pytesseract      | 0.3.10    |
| scikit-learn     | 1.2.2     |
| opencv-python    | 4.7.0.72  |

## Use
### ```basicPipeline.py```
Basic project pipeline to get two separate csv file outputs used in later testing.
Here you will only need to change the file name for ``output_file`` and ``full_text`` every time to to make changes to preprocessing to save new results to a different file if you want to retain the results of previous tests. ``input_folder`` already has randomly selected sample images from [kaggle](https://www.kaggle.com/datasets/shensivam/nutritional-facts-from-food-label) unless you want to add your own, now you can just run the code.
```python
input_folder = "./sample_images"
output_file = "./nutrivaluesfrompipeline/output.csv"
full_text = "./fulltextfrompipeline/fulltext.csv"
hard = pd.read_csv("./hardcodednutrilabels/hardfulltext.csv")
```
To make changes/adjustments to image preprocessing you will need to navigate to ``preprocessing.py`` and find function called ``preprocess_image``
```python
def preprocess_image(image):
    # Resize the image to a fixed size of 400x400 pixels
    image = cv.resize(image, (400, 400))
    gray = adaptive_gamma_correction(image)
    # Apply Binary Inverse thresholding to the grayscale image to obtain a binary image.
    thresh = cv.threshold(gray, 120, 255, cv.THRESH_BINARY_INV)[1]
    # Return the preprocessed binary image.
    return thresh
```
Here you can add or adjust variables for different image preprocessing techniques that will pressent different outcomes when running ``basicPipeLine.py``
To specify what exactly you want to extract from the image you have to adjust the regex patterns inside ``extract_nutritional_info`` function found in ``preprocessing.py``. Currently they are set to **Calories**, **Fat**, **Carbohydrate** and **Protein**. If you make any changes you will also need to adjust the code inside ``process_images`` function to reflect those changes inside csv file.
### ```compareCharCount.py```
```python
hard = "./hardcodednutrilabels/hard_nutri.csv"
sample = "./nutrivaluesfrompipeline/output_v2_1.csv"
result = prep.compare_csv_files(hard, sample)
prep.print_comparison_results(result)
```
``compareCharCount.py`` uses **'your_filename.csv'** ``output_file = "./nutrivaluesfrompipeline/output.csv"`` in this case **output_v2_1.csv** that was created when you ran ``basicPipeLine.py`` and compares it against hard coded values inside ``./hardcodednutrilabels/hard_nutri.csv``, returning character count for each image and accuracy. Data was taken from the sample images found in the **sample_images** directory, so if youy want to change the images to test you will also have to adjust both ``hard_nutri.csv`` and ``hardfulltext.csv`` to get propper test results.
### ```compareSMAPE.py```
```python
hard = "./hardcodednutrilabels/hardfulltext.csv"
sample = "./fulltextfrompipeline/fulltext_v1_4.csv"
prep.compare_individual_csv_files(hard, sample)
```
``compareSMAPE.py`` very similarly to ``compareCharCount.py`` compares two csv files except now it uses the other csv file made from running our ``basicPipeline.py`` ``full_text = "./fulltextfrompipeline/fulltext.csv"`` in our case is **fulltext_v1_4.csv** and is compared up against **hardfulltext.csv** which is again hardcoded data set from our sample images. This returns the occurance of each individual character in the two csv files and calculates individual and and overall **SMAPE**(Symmetric mean absolute percentage error) and **MAE**(Mean absolute error) score, SMAPE being more accurate accuracy scoring method thus we will be focusing mostly on it.
### ```findBestPerformance.py```
This python file contains a function to find best variables in this case ``bestBlockSize`` and ``bestC`` for an OpenCV function
```python
cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize, C)
```
In short the function loops through all sample images inside ``./sample_images`` peforming similar image preprocessing as ``basicPipeLine.py``, calculating average SMAPE and storing it and current variables used from our variable range before moving onto the next value in our set range.
```python
# Define the range of values to test for blockSize and C 
blockSizes = range(11, 61, 2) # odd numbers
Cs = range(0, 31, 2)
```
Finally at the end it displays heatmap with the variables and coresponding average SMAPE score so that we can analize and choose the best parameters.
<img alt="Heatmap" src="https://github.com/M-Strzelecki/OCRResearch/blob/updatingPipeline/heatmapdata/smape_heatmap%26timetaken.png"/>
Currently they way function is set-up it takes about 40 minutes to run the test, this time will either decrease or increase depending on the ammount of preprocessing steps done (can add more) to the images, the range of ``blockSizes`` and ``Cs`` (you can adjust) and the ammount of images inside ``./sample_images`` (if changed you need to remember like in previous functions the hard coded data set will also need adjustment). 
This function is still very un-polished but could be used to test other image preprocessing fuctions from OpenCV library.
### Other Python Files
Other **py** files such as ``isImageBimodal.py``, ``gammaExample.py``, ``otsuThresholdingTest.py`` and ``testingGammaThresholding`` were created for research or presentation purposes to get better visuals of different types of preprocesses and the differences when we adjust their variables. Lastly ``autoGamma.py`` was created to test different methods/ways I could adaptively adjust gamma levels, no mater what image was loaded in it wouldent be over or under exposed. All of these python files are quite simple and can be easily adjusted as the code is commented. Simply run as is to see the results and then enjoy changing the variables to get differnet outcomes.
## Future Work/Plans
As this was my first big research I made many mistakes that I have learned on. I really enjoyed the topis and plan on carrying it on in the near future to research and create new functions:
   *New ways to preprocess images to increase OCR perofrmance
   *More function to automate the choice of variables in OpenCV's image preprocessing functions
   *Research ways to determine if image quality is to low to run the OCR engine on
   *Create more advanced pipe line
## License
Copyright (c) 2023, Michal Strzelecki, Munster Technological University
<br>All rights reserved.
