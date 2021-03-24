# Langage detection and machine translation
All journal articles and their metadata are organized into a Python dataframe. To identify the non-English articles, we used three automatic language-detection methods (language metadata when available in the downloaded documents, langid and langdetect packages) and extracted all documents for which non-English was detected at least once.
