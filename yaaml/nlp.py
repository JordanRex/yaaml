#### NLP MODULE ####

"""
Description - Creates NLP features from the text columns in data (if any) and adds them to the main model (not automated, user discretion). Makes a large number of word features (count, term frequency and hashed), performs PCA on them to reduce the dimension, and creates a separate model to validate the predictive power in the text features. If results are better than the baseline, an ensemble model is automatically created. the user is advised to experiment by adding the features to the main model and compare with the ensemble results. if the results are around the ballpark of the ensemble results, it is highly recommended to use them as features instead of as a separate model in an ensemble
"""

