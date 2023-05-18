import preprocessing as prep

"""
Compare two csv files and return the difference between them in character count and their accuracy
"""

hard = "./hardcodednutrilabels/hard_nutri.csv"
sample = "./nutrivaluesfrompipeline/output_v2_1.csv"

result = prep.compare_csv_files(hard, sample)
prep.print_comparison_results(result)
# print(f"Results: {result}")
