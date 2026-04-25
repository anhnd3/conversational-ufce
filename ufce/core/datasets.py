
# This script would be utilised to enhance the experiment on multiple data sets.
# Also, it could be integrated with the interactive elements for the default user preferences.

class DATSETS():
    def __init__(self):
        self.datasets_name = ['bank', 'credit', 'adult']

    @staticmethod
    def credit_details():
        features = ['Married', 'Single', 'Age_lt_25',
               'Age_in_25_to_40', 'Age_in_40_to_59', 'Age_geq_60', 'EducationLevel',
               'MaxBillAmountOverLast6Months', 'MaxPaymentAmountOverLast6Months',
               'MonthsWithZeroBalanceOverLast6Months',
               'MonthsWithLowSpendingOverLast6Months',
               'MonthsWithHighSpendingOverLast6Months', 'MostRecentBillAmount',
               'MostRecentPaymentAmount', 'TotalOverdueCounts', 'TotalMonthsOverdue',
               'HistoryOfOverduePayments']
        user_feature_list = ['MaxBillAmountOverLast6Months', 'MaxPaymentAmountOverLast6Months', 'MostRecentBillAmount',
               'MostRecentPaymentAmount', 'TotalMonthsOverdue']
        feature_flags = {'MaxBillAmountOverLast6Months':'A', 'MaxPaymentAmountOverLast6Months':'A', 'MostRecentBillAmount':'A',
               'MostRecentPaymentAmount':'A', 'TotalMonthsOverdue':'A'}
        threshold_values = {'MaxBillAmountOverLast6Months':5000, 'MaxPaymentAmountOverLast6Months':10000, 'MostRecentBillAmount':5000,
               'MostRecentPaymentAmount':5000, 'TotalMonthsOverdue':5}
        order_of_asymmetric= {'MaxBillAmountOverLast6Months':'I', 'MaxPaymentAmountOverLast6Months':'I', 'MostRecentBillAmount':'I',
               'MostRecentPaymentAmount':'I', 'TotalMonthsOverdue':'I'}
        protected_features = []
        features_for_corr = ['MaxBillAmountOverLast6Months', 'MaxPaymentAmountOverLast6Months', 'MostRecentBillAmount',
               'MostRecentPaymentAmount', 'TotalMonthsOverdue']
        perturbing_rates = {'MaxBillAmountOverLast6Months':50, 'MaxPaymentAmountOverLast6Months':100, 'MostRecentBillAmount':500,
               'MostRecentPaymentAmount':500, 'TotalMonthsOverdue':1}
        u_f_cat_list = ['Married', 'Single', 'Age_lt_25', 'Age_in_25_to_40', 'Age_in_40_to_59', 'Age_geq_60']
        return features, user_feature_list, feature_flags, threshold_values, order_of_asymmetric, protected_features, features_for_corr, perturbing_rates, u_f_cat_list

    @staticmethod
    def bank_details():
        features = ['Income', 'Family', 'CCAvg', 'Education', 'Mortgage', 'SecuritiesAccount', 'CDAccount', 'Online', 'CreditCard']
        user_feature_list = ['Income', 'Mortgage', 'CCAvg', 'Education']
        feature_flags = {'Income': 'A', 'Mortgage': 'A' , 'CCAvg': 'A', 'Education':'A'}
        threshold_values = {'Income': 100, 'Mortgage': 50, 'CCAvg':0.6, 'Education': 1.0}
        order_of_asymmetric= {'Income': 'I', 'Mortgage': 'I', 'CCAvg': 'I', 'Education': 'I'}
        protected_features = []
        features_for_corr = ['Income', 'CCAvg', 'Mortgage', 'Education']
        perturbing_rates = {'Income': 1.0, 'Mortgage': 1, 'CCAvg':0.1, 'Education': 1.0}
        u_f_cat_list = []
        return features, user_feature_list, feature_flags, threshold_values, order_of_asymmetric, protected_features, features_for_corr, perturbing_rates, u_f_cat_list

    @staticmethod
    def adult_details():
        features = ['workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','hours-per-week','native-country','net_capital','per-hour-income']
        user_feature_list = ['workclass', 'education', 'net_capital', 'per-hour-income']
        feature_flags = {'workclass':'A', 'education':'A', 'net_capital':'A', 'per-hour-income':'A'}
        threshold_values = {'workclass':15, 'education':5, 'net_capital':1000, 'per-hour-income':5}
        order_of_asymmetric= {'workclass':'I', 'education':'I', 'net_capital':'I', 'per-hour-income':'I'}
        protected_features = []
        features_for_corr = ['workclass', 'education', 'net_capital', 'per-hour-income']
        perturbing_rates = {'workclass': 2.0, 'fnlwgt':100, 'education':1.0, 'education-num':1.0, 'marital-status':1.0, 'occupation':1.0, 'relationship':1.0, 'hours-per-week':1.0,'native-country':1.0, 'net_capital':50, 'per-hour-income':0.5 }
        u_f_cat_list = []
        return features, user_feature_list, feature_flags, threshold_values, order_of_asymmetric, protected_features, features_for_corr,perturbing_rates, u_f_cat_list

    @staticmethod
    def wine_details():

        features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
               'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
               'pH', 'sulphates', 'alcohol']
        user_feature_list = ['fixed acidity', 'residual sugar', 'free sulfur dioxide', 'total sulfur dioxide', 'pH', 'alcohol', 'density', 'volatile acidity']
        feature_flags = {'fixed acidity':'A', 'residual sugar':'A', 'free sulfur dioxide':'A', 'total sulfur dioxide':'A', 'pH':'A', 'alcohol':'A', 'density':'A', 'volatile acidity':'A'}
        threshold_values = {'fixed acidity':3.0, 'residual sugar':3.0, 'free sulfur dioxide':8.0, 'total sulfur dioxide':12.0, 'pH':1.0, 'alcohol':2.0, 'density':0.20, 'volatile acidity':0.20}
        order_of_asymmetric= {'fixed acidity':'I', 'residual sugar':'I', 'free sulfur dioxide':'I', 'total sulfur dioxide':'I', 'pH':'I', 'alcohol':'I', 'density':'I', 'volatile acidity':'I'}
        protected_features = []
        features_for_corr = ['fixed acidity', 'volatile acidity', 'residual sugar', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
               'pH', 'alcohol']
        perturbing_rates = {'fixed acidity':0.5, 'residual sugar':0.5, 'free sulfur dioxide':0.5, 'total sulfur dioxide':1, 'pH':0.2, 'alcohol':0.5, 'density':0.05, 'volatile acidity':0.05}
        u_f_cat_list = []
        return features, user_feature_list, feature_flags, threshold_values, order_of_asymmetric, protected_features, features_for_corr, perturbing_rates, u_f_cat_list

    @staticmethod
    def wisconsin_details():
        features = ['ClumpThickness', 'CellSize', 'CellShape', 'MarginalAdhesion', 'EpithelialSize', 'BareNuclei', 'BlandChromatin', 'NormalNucleoli', 'Mitoses']
        user_feature_list = ['CellSize', 'CellShape','EpithelialSize', 'BareNuclei', 'NormalNucleoli', 'MarginalAdhesion']
        feature_flags = {'CellSize':'A', 'CellShape':'A', 'EpithelialSize':'A', 'BareNuclei':'A', 'NormalNucleoli':'A', 'MarginalAdhesion':'A'}
        threshold_values = {'CellSize':6, 'CellShape':6, 'EpithelialSize':6, 'BareNuclei':6, 'NormalNucleoli':6, 'MarginalAdhesion':6}
        order_of_asymmetric= {'CellSize':'D', 'CellShape':'D', 'EpithelialSize':'D', 'BareNuclei':'D', 'NormalNucleoli':'D', 'MarginalAdhesion':'D'}
        protected_features = []
        features_for_corr = ['CellSize', 'CellShape','EpithelialSize', 'BareNuclei', 'NormalNucleoli', 'MarginalAdhesion']
        perturbing_rates = {'CellSize':0.3, 'CellShape':0.3, 'EpithelialSize':0.3, 'BareNuclei':0.3, 'NormalNucleoli':0.3, 'MarginalAdhesion':0.3}
        u_f_cat_list = []
        return features, user_feature_list, feature_flags, threshold_values, order_of_asymmetric, protected_features, features_for_corr, perturbing_rates, u_f_cat_list

    @staticmethod
    def bupa_details():
        #bupa liver disorder either alcoholism-1 or not-2, to make it
        features = ['Mcv', 'Alkphos', 'Sgpt', 'Sgot', 'Gammagt', 'Drinks']
        user_feature_list = ['Sgpt', 'Sgot', 'Gammagt']
        feature_flags = {'Sgpt':'A', 'Sgot':'A', 'Gammagt':'A'}
        threshold_values = {'Sgpt':15, 'Sgot':15, 'Gammagt':15}
        order_of_asymmetric= {'Sgpt':'I', 'Sgot':'I', 'Gammagt':'I'}
        protected_features = []
        features_for_corr = ['Sgpt', 'Sgot', 'Gammagt', 'Drinks']
        perturbing_rates = {'Sgpt':1, 'Sgot':1, 'Gammagt':1}
        u_f_cat_list = []
        return features, user_feature_list, feature_flags, threshold_values, order_of_asymmetric, protected_features, features_for_corr, perturbing_rates, u_f_cat_list

    @staticmethod
    def appendis_details():
        #appendicitis  , appendicitis-1 , not-appendicitis-0
        features = ['At1', 'At2', 'At3','At6', 'At7']
        user_feature_list = ['At1', 'At2', 'At3', 'At6', 'At7']
        feature_flags = {'At1':'A', 'At2':'A', 'At3':'A','At6':'A', 'At7':'A'}
        threshold_values = {'At1':0.8, 'At2':0.8, 'At3':0.8,'At6':0.8, 'At7':0.8}
        order_of_asymmetric= {'At1':'I', 'At2':'I', 'At3':'I','At6':'I', 'At7':'I'}
        protected_features = []
        features_for_corr = ['At1', 'At2', 'At3', 'At6', 'At7']
        perturbing_rates = {'At1':0.1, 'At2':0.1, 'At3':0.1, 'At6':0.1, 'At7':0.1}
        u_f_cat_list = []
        return features, user_feature_list, feature_flags, threshold_values, order_of_asymmetric, protected_features, features_for_corr, perturbing_rates, u_f_cat_list

    @staticmethod
    def saheart_details():
        #saheart, 0 represents no 1 means there is coronary disease
        features = ['Sbp', 'Tobacco', 'Ldl', 'Adiposity', 'Famhist', 'Typea', 'Obesity',
               'Alcohol', 'Age']
        user_feature_list = ['Sbp','Adiposity', 'Obesity', 'Ldl']
        feature_flags = {'Sbp':'A','Adiposity':'A', 'Obesity':'A', 'Ldl':'A'}
        threshold_values = {'Sbp':25,'Adiposity':15, 'Obesity':10, 'Ldl':5}
        order_of_asymmetric= {'Sbp':'I','Adiposity':'I', 'Obesity':'I', 'Ldl':'I'}
        protected_features = []
        features_for_corr = ['Sbp','Adiposity', 'Obesity', 'Ldl']
        perturbing_rates = {'Sbp':0.5,'Adiposity':0.5, 'Obesity':0.5, 'Ldl':0.5}
        u_f_cat_list = []
        return features, user_feature_list, feature_flags, threshold_values, order_of_asymmetric, protected_features, features_for_corr,perturbing_rates, u_f_cat_list

    @staticmethod
    def spotify_details():
        features = ['acousticness', 'danceability', 'duration_ms', 'energy',
               'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
               'speechiness', 'tempo', 'time_signature', 'valence']
        user_feature_list = ['acousticness', 'danceability', 'energy', 'loudness']
        feature_flags = {'acousticness':'A', 'danceability':'A', 'energy':'A', 'loudness':'A'}
        threshold_values = {'acousticness':0.5, 'danceability':0.5, 'energy':0.5, 'loudness':5.0}
        order_of_asymmetric= {'acousticness':'I', 'danceability':'I', 'energy':'D', 'loudness':'I'}
        protected_features = []
        features_for_corr = ['acousticness', 'danceability', 'energy', 'loudness']
        perturbing_rates = {'acousticness':0.05, 'danceability':0.01, 'energy':0.01, 'loudness':0.5}
        u_f_cat_list = []
        return features, user_feature_list, feature_flags, threshold_values, order_of_asymmetric, protected_features, features_for_corr, perturbing_rates, u_f_cat_list

    @staticmethod
    def movie_details():
        features = ['Marketing expense', 'Production expense', 'Multiplex coverage',
               'Budget', 'Movie_length', 'Lead_ Actor_Rating', 'Lead_Actress_rating',
               'Director_rating', 'Producer_rating', 'Critic_rating', 'Trailer_views',
               '3D_available', 'Time_taken', 'Twitter_hastags', 'Genre',
               'Avg_age_actors', 'Num_multiplex', 'Collection']
        user_feature_list = ['Production expense', 'Multiplex coverage','Num_multiplex', 'Movie_length', 'Lead_ Actor_Rating', 'Lead_Actress_rating',
               'Director_rating', 'Producer_rating', 'Genre', 'Collection']
        feature_flags = {'Production expense':'A', 'Num_multiplex':'A', 'Multiplex coverage':'A', 'Movie_length':'A', 'Lead_ Actor_Rating':'A', 'Lead_Actress_rating':'A',
               'Director_rating':'A', 'Producer_rating':'A', 'Genre':'S', 'Collection':'A'}
        threshold_values = {'Production expense': 40, 'Num_multiplex':50,'Multiplex coverage':0.4, 'Movie_length':35, 'Lead_ Actor_Rating':4.0, 'Lead_Actress_rating':4.0,
               'Director_rating':4.0, 'Producer_rating':4.0, 'Genre':3, 'Collection':20000}
        order_of_asymmetric= {'Production expense':'I', 'Multiplex coverage':'I','Num_multiplex':'I', 'Movie_length':'I', 'Lead_ Actor_Rating':'I', 'Lead_Actress_rating':'I',
               'Director_rating':'I', 'Producer_rating':'I', 'Genre':'I', 'Collection':'I'}
        protected_features = []
        features_for_corr = ['Production expense', 'Multiplex coverage','Num_multiplex', 'Movie_length', 'Lead_ Actor_Rating', 'Lead_Actress_rating',
               'Director_rating', 'Producer_rating', 'Genre', 'Collection']
        perturbing_rates = {'Production expense': 1.0, 'Multiplex coverage':0.1, 'Num_multiplex':1.0, 'Movie_length':1.0, 'Lead_ Actor_Rating':0.4, 'Lead_Actress_rating':0.4,
               'Director_rating':0.4, 'Producer_rating':0.4, 'Genre':1, 'Collection':500}
        u_f_cat_list = []
        return features, user_feature_list, feature_flags, threshold_values, order_of_asymmetric, protected_features, features_for_corr, perturbing_rates, u_f_cat_list

    @staticmethod
    def heart_details():
        features = ['ChestPainType', 'RestBloodPressure', 'SerumCholestoral',
               'FastingBloodSugar', 'ResElectrocardiographic', 'MaxHeartRate',
               'ExerciseInduced', 'Oldpeak', 'MajorVessels', 'Thal']
        user_feature_list = ['ChestPainType', 'MaxHeartRate', 'ExerciseInduced']
        feature_flags = {'ChestPainType':'A', 'MaxHeartRate':'S', 'ExerciseInduced':'A'}
        threshold_values = {'ChestPainType':2, 'MaxHeartRate':15, 'ExerciseInduced':1}
        order_of_asymmetric= {'ChestPainType':'D', 'MaxHeartRate':'I', 'ExerciseInduced':'I', 'Oldpeak':'I', 'Slope':'I'}
        protected_features = []
        features_for_corr = ['ChestPainType', 'MaxHeartRate','ExerciseInduced']
        perturbing_rates = {'ChestPainType':1, 'MaxHeartRate':1, 'ExerciseInduced':1, 'Oldpeak':0.1, 'Slope':1}
        u_f_cat_list = []
        return features, user_feature_list, feature_flags, threshold_values, order_of_asymmetric, protected_features, features_for_corr, perturbing_rates, u_f_cat_list

    @staticmethod
    def magic_details():
        features = ['FLength', 'FWidth', 'FSize', 'FConc', 'FConc1', 'FAsym', 'FM3Long',
               'FM3Trans', 'FAlpha', 'FDist']
        user_feature_list = ['FLength', 'FWidth', 'FSize', 'FConc', 'FConc1']
        feature_flags = {'FLength':'A', 'FWidth':'A', 'FSize':'A', 'FConc':'A', 'FConc1':'A'}
        threshold_values = {'FLength':30.0, 'FWidth':20.0, 'FSize':2.0, 'FConc':0.5, 'FConc1':0.2}
        order_of_asymmetric= {'FLength':'I', 'FWidth':'I', 'FSize':'I', 'FConc':'I', 'FConc1':'I'}
        protected_features = []
        features_for_corr = ['FLength', 'FWidth', 'FSize', 'FConc', 'FConc1']
        perturbing_rates = {'FLength':1.0, 'FWidth':1.0, 'FSize':0.5, 'FConc':0.2, 'FConc1':0.1} #Fsize:0.1, Fcon1:0.05
        u_f_cat_list = []
        return features, user_feature_list, feature_flags, threshold_values, order_of_asymmetric, protected_features, features_for_corr, perturbing_rates, u_f_cat_list

    @staticmethod
    def wdbc_details():
        features = ['Radius1', 'Texture1', 'Perimeter1', 'Area1', 'Smoothness1',
               'Compactness1', 'Concavity1', 'Concave_points1', 'Symmetry1',
               'Fractal_dimension1', 'Radius2', 'Texture2', 'Perimeter2', 'Area2',
               'Smoothness2', 'Compactness2', 'Concavity2', 'Concave_points2',
               'Symmetry2', 'Fractal_dimension2', 'Radius3', 'Texture3', 'Perimeter3',
               'Area3', 'Smoothness3', 'Compactness3', 'Concavity3', 'Concave_points3',
               'Symmetry3', 'Fractal_dimension3']
        user_feature_list = ['Radius1', 'Perimeter1', 'Area3', 'Perimeter3']
        feature_flags = {'Radius1':'A', 'Perimeter1':'A', 'Area3':'A', 'Perimeter3':'A'}
        threshold_values = {'Radius1':6.0, 'Perimeter1':15.0, 'Area3':200.0, 'Perimeter3':20}
        order_of_asymmetric= {'Radius1':'I', 'Perimeter1':'I', 'Area3':'I', 'Perimeter3':'I'}
        protected_features = []
        features_for_corr = ['Radius1', 'Perimeter1', 'Area3', 'Perimeter3']
        perturbing_rates = {'Radius1':0.5, 'Perimeter1':1.0, 'Area3':10.0, 'Perimeter3':1}
        u_f_cat_list = []
        return features, user_feature_list, feature_flags, threshold_values, order_of_asymmetric, protected_features, features_for_corr, perturbing_rates, u_f_cat_list

    @staticmethod
    def titanic_details():
        features = ['Class', 'Age', 'Sex']
        user_feature_list = ['Class', 'Age', 'Sex']
        feature_flags = {'Class':'A', 'Age':'A', 'Sex':'A'}
        threshold_values = {'Class':2.0, 'Age':1.0, 'Sex':1.0}
        order_of_asymmetric= {'Class':'I', 'Age':'I', 'Sex':'I'}
        protected_features = []
        features_for_corr = ['Class', 'Age', 'Sex']
        perturbing_rates = {'Class':1.0, 'Age':1.0, 'Sex':1.0}
        u_f_cat_list = ['Age', 'Sex']
        return features, user_feature_list, feature_flags, threshold_values, order_of_asymmetric, protected_features, features_for_corr, perturbing_rates, u_f_cat_list

    @staticmethod
    def mammograpic_details():
        features = ['BI-RADS', 'Age', 'Shape', 'Margin', 'Density']
        user_feature_list = ['BI-RADS', 'Age', 'Shape', 'Margin']
        feature_flags = {'BI-RADS':'A', 'Age':'A', 'Shape':'A', 'Margin':'A'}
        threshold_values = {'BI-RADS':5, 'Age':15, 'Shape':2, 'Margin':3}
        order_of_asymmetric= {'BI-RADS':'D', 'Age':'D', 'Shape':'D', 'Margin':'D'}
        protected_features = []
        features_for_corr = ['BI-RADS','Age', 'Shape', 'Margin']
        perturbing_rates = {'BI-RADS':1, 'Age':1, 'Shape':1, 'Margin':1}
        u_f_cat_list = []
        return features, user_feature_list, feature_flags, threshold_values, order_of_asymmetric, protected_features, features_for_corr, perturbing_rates, u_f_cat_list

