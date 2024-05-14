import joblib
import numpy as np

patients = ['540', '544', '552', '567', '584', '596', 'allsubs']
year = '2020'

for patient in patients:
    adversarial_data = joblib.load('/Data/Patients/' + year + '/' + patient + '/adversarial_data.pkl')
    benign_data = joblib.load('/Data/Patients/' + year + '/' + patient + '/benign_data.pkl')
    predicted_output = np.array(joblib.load('/Data/Patients/' + year + '/' + patient + '/predicted_output.pkl'))
    actual_output = np.array(joblib.load('/Data/Patients/' + year + '/' + patient + '/actual_output.pkl'))

    coefficient = np.empty([actual_output.shape[0], actual_output.shape[1]])
    magnitude = np.empty([actual_output.shape[0], actual_output.shape[1]])
    instantaneous_error = np.empty([actual_output.shape[0], actual_output.shape[1]])
    for i in range(len(actual_output)):
        postprandial = any([benign_data[i][0][7], benign_data[i][1][7], benign_data[i][2][7], benign_data[i][3][7],
                benign_data[i][4][7], benign_data[i][5][7], benign_data[i][6][7], benign_data[i][7][7],
                benign_data[i][8][7], benign_data[i][9][7], benign_data[i][10][7], benign_data[i][11][7]]) #check if postprandial (True) or fasting (False)
        for j in range(len(actual_output[i])):
            if not postprandial: #fasting
                if actual_output[i][j] < 70 and predicted_output[i][j] > 125:   # actual (hypo), predicted (hyper)
                    coefficient[i][j] = 64
                elif 70 < actual_output[i][j] < 125 < predicted_output[i][j]:   # actual (normal), predicted (hyper)
                    coefficient[i][j] = 32
                elif actual_output[i][j] < 70 < predicted_output[i][j] < 125:   # actual (hypo), predicted (normal)
                    coefficient[i][j] = 16
                elif actual_output[i][j] > 125 and predicted_output[i][j] < 70: # actual (hyper), predicted (hypo)
                    coefficient[i][j] = 8
                elif actual_output[i][j] > 125 > predicted_output[i][j] > 70:   # actual (hyper), predicted (normal)
                    coefficient[i][j] = 4
                elif 125 > actual_output[i][j] > 70 > predicted_output[i][j]:   # actual (normal), predicted (hypo)
                    coefficient[i][j] = 2
            else:   #postprandial
                if actual_output[i][j] < 70 and predicted_output[i][j] > 180:   # actual (hypo), predicted (hyper)
                    coefficient[i][j] = 64
                elif 70 < actual_output[i][j] < 180 < predicted_output[i][j]:   # actual (normal), predicted (hyper)
                    coefficient[i][j] = 32
                elif actual_output[i][j] < 70 < predicted_output[i][j] < 180:   # actual (hypo), predicted (normal)
                    coefficient[i][j] = 16
                elif actual_output[i][j] > 180 and predicted_output[i][j] < 70: # actual (hyper), predicted (hypo)
                    coefficient[i][j] = 8
                elif actual_output[i][j] > 180 > predicted_output[i][j] > 70:   # actual (hyper), predicted (normal)
                    coefficient[i][j] = 4
                elif 180 > actual_output[i][j] > 70 > predicted_output[i][j]:   # actual (normal), predicted (hypo)
                    coefficient[i][j] = 2

            magnitude[i][j] = pow(predicted_output[i][j] - actual_output[i][j], 2)
            instantaneous_error[i][j] = coefficient[i][j] * magnitude[i][j]

    print(instantaneous_error.shape)
    joblib.dump(instantaneous_error,'/Data/Patients/' + year + '/' + patient + '/instantaneous_error.pkl')

