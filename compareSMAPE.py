import preprocessing as prep

"""
Compare two csv files and return a string representation of the difference between them and their corresponding MAE and SMAPE
"""

hard = "./hardcodednutrilabels/hardfulltext.csv"
sample = "./fulltextfrompipeline/fulltext_v1_4.csv"

result = prep.count_chars_in_file("./hardcodednutrilabels/hardfulltext.csv")
# prep.print_individual_count(result)

result2 = prep.count_chars_in_file("./fulltextfrompipeline/fulltext.csv")
# prep.print_individual_count(result2)

prep.compare_individual_csv_files(hard, sample)
