import joblib
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import numpy

patients = ['540', '544', '552', '567', '584', '596', 'allsubs']


patient = patients[0]
df_0 = joblib.load('/Data/' + patient + '/instantaneous_error.pkl')
patient = patients[1]
df_1 = joblib.load('/Data/' + patient + '/instantaneous_error.pkl')
patient = patients[2]
df_2 = joblib.load('/Data/' + patient + '/instantaneous_error.pkl')
patient = patients[3]
df_3 = joblib.load('/Data/' + patient + '/instantaneous_error.pkl')
patient = patients[4]
df_4 = joblib.load('/Data/' + patient + '/instantaneous_error.pkl')
patient = patients[5]
df_5 = joblib.load('/Data/' + patient + '/instantaneous_error.pkl')

size = min(len(df_0), len(df_1), len(df_2), len(df_3), len(df_4), len(df_5))
# df = numpy.array([df_0.to_numpy()[:size,:], df_1.to_numpy()[:size,:], df_2.to_numpy()[:size,:], df_3.to_numpy()[:size,:], df_4.to_numpy()[:size,:], df_5.to_numpy()[:size,:]])
df = numpy.array([df_0[:size], df_1[:size], df_2[:size], df_3[:size], df_4[:size], df_5[:size]])
# numpy.random.shuffle(df)
# print(df)

# Keep only 50 time series
X_train = TimeSeriesScalerMeanVariance().fit_transform(df) #df[:50]
# Make time series shorter
# X_train = TimeSeriesResampler(sz=40).fit_transform(X_train)
sz = X_train.shape[1]


seeds = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 1000]
for seed in seeds:
    numpy.random.seed(seed)
    # DBA-k-means
    # print("DBA k-means")
    dba_km = TimeSeriesKMeans(n_clusters=2,
                              n_init=100,
                              metric="dtw",
                              verbose=False,
                              max_iter_barycenter=100,
                              random_state=seed)
    y_pred = dba_km.fit_predict(X_train)

    print('seed:' + str(seed))
    print('labels:')
    print(dba_km.labels_)
    print('------------------------------------------')
    print('inertia:')
    print(dba_km.inertia_)
    print('------------------------------------------')
