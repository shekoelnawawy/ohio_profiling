benign_data: Original data before manipulation
Shape - (x, 12, 8) x=number of observations...12=number of points back in time used for prediction...8= number of features
Features in order - CGM...Bolus dose...carbs...time (encoded as sin)...time (encoded as cos)...finger glucose...CGM missing?...postprandial?
postprandial indicates whether this observation falls within two hours of carb intake.

adversarial_data: Adversarial data after manipulation
Shape - (x, 12, 7) x=number of observations...12=number of points back in time used for prediction...7= number of features
Features in order - CGM...Bolus dose...carbs...time (encoded as sin)...time (encoded as cos)...finger glucose...CGM missing?
postprandial does not exist here for compatibility with the used Ohio model but it can be the same as benign data since it only indicates whether this observation falls within two hours of carb intake.

actual_output: Benign predictions
Shape - (x, 6) x=number of observations...6-number of future predictions

predicted_output: Compromised predictions
Shape - (x, 6) x=number of observations...6-number of future predictions

instantaneous_error: error calculated using the discussed formula
Shape - (x, 6) x=number of observations...6-number of calculated errors
