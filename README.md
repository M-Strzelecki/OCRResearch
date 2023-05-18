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
### basicPipeline.py
```python
input_folder = "./sample_images"
output_file = "./nutrivaluesfrompipeline/output.csv"
full_text = "./fulltextfrompipeline/fulltext.csv"
hard = pd.read_csv("./hardcodednutrilabels/hardfulltext.csv")
```