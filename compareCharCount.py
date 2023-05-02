import preprocessing as prep

hard = "hard_nutri.csv"
sample = "output_v2_1.csv"

result = prep.compare_csv_files(hard, sample)
prep.print_comparison_results(result)
# print(f"Results: {result}")
