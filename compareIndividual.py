import preprocessing as prep

hard = "hardfulltext.csv"
sample = "fulltext.csv"

result = prep.count_chars_in_file("hardfulltext.csv")
# prep.print_individual_count(result)

result2 = prep.count_chars_in_file("fulltext.csv")
# prep.print_individual_count(result2)

prep.compare_individual_csv_files(hard, sample)
