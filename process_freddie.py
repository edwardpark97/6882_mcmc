import csv
import numpy as np
import os


'''	
Dependent Variable:
whether a loan was foreclosed by the end of september 2017
	go into time-historical data,
	if column 9 is "03" then was foreclosed 

Predictor Variables (NOT DISCRETE):
FICO score (column 1)
Date of first mortgage payment (column 2)
Maturity date (column 4)
Mortgage Interest percentage (column 6)
Combined loan to value ratio (column 9)
Debt to income ratio (column 10)
Original principal balance of the loan (column 11)
Original loan to value ratio (column 12)
Original interest rate (column 13)

Predictor Variables (DISCRETE):
Date of first mortgage payment (column 2)
FICO score (column 1)
Debt to income ratio (column 10)
Original principal balance of the loan (column 11)
First-time home-buyer status (column 3) - "yes", "no", "unknown"
'''

def YYYYMM_to_month(date):
	return 12 * (int(date[:4]) - 2009) + int(date[4:])

def create_row_features(r):
	def YYYYMM_to_month(date):
		return 12 * (int(date[:4]) - 2009) + int(date[4:])
	x = [r[0],					# 0 FICO score
		YYYYMM_to_month(r[1]),	# 1 date of first mortgage payment
		YYYYMM_to_month(r[3]),	# 2 maturity date
		r[5],					# 3 mortgage interest percentage
		r[8],					# 4 combined loan-to-value ratio
		r[9],					# 5 debt-to-income ratio
		r[10],					# 6 original principal balance of the loan
		r[11],					# 7 original loan-to-value ratio
		r[12]]					# 8 original interest rate
	return np.array(x)

# PROBABLY NOT USED ANYMORE
def create_row_features_discrete(r):
	x = [YYYYMM_to_month(r[1]),
		r[0],
		r[9],
		r[10],
		int(r[2] == 'Y'),
		int(r[2] == 'N'),
		int(r[2] != 'Y' and r[2] != 'N')]
	return np.array(x)

def create_arrays(use_discrete_vars=False):
	# Try loading the numpy array
	if os.path.isfile("freddie_mac/np_arrays.npz"):
		print("Loading from local")
		np_array = np.load("freddie_mac/np_arrays.npz")
		return np_array['feature_array'], np_array['output_vector']
	else:
		try:
			import boto3
			import botocore
			ACCESS_ID = 'ACCESS_ID_HERE'
			ACCESS_KEY = 'ACCESS_KEY_HERE'
			AWS_BUCKET = '6882-mcmc'
			FREDDIE_MAC_FILEPATH = 'freddie_mac_arrays.npz'
			s3 = boto3.resource('s3', 
				aws_access_key_id=ACCESS_ID,
				aws_secret_access_key=ACCESS_KEY)
			s3.Bucket(AWS_BUCKET).download_file(FREDDIE_MAC_FILEPATH, 'freddie_mac/np_arrays.npz')
		except Exception:
			print("Tried to download from AWS bucket, didn't work")
			pass

	q1, q2, q3, q4 = 587250, 654170, 381904, 350521 # num data points in each file
	num_data_points = q1 + q2 + q3 + q4
	num_features = 7 if use_discrete_vars else 9 #number of columns from above 

	feature_array = np.zeros((num_data_points, num_features))
	output_vector = np.zeros(num_data_points)
	index_to_loan_sequence = np.zeros(num_data_points) # maps the numpy index to the loan sequence number

	# generate feature_array
	row_count = 0
	for i in range(1, 5):
		with open("freddie_mac/historical_data1_Q%s2009.txt" % i) as csvfile:
			reader = csv.reader(csvfile, delimiter="|")
			for row in reader:
				feature_array[row_count] = create_row_features_discrete(row) if use_discrete_vars \
											else create_row_features(row)
				index_to_loan_sequence[row_count] = int(row[19][5:])
				row_count += 1

	# generate output_vector
	row_count = 0
	for i in range(1, 5):
		with open("freddie_mac/historical_data1_time_Q%s2009.txt" % i) as csvfile:
			reader = csv.reader(csvfile, delimiter="|")
			for row in reader:
				while index_to_loan_sequence[row_count] != int(row[0][5:]):
					row_count += 1
				if row[8] == '':
					continue
				if row[8] == '03':
					output_vector[row_count] = 1

	# norm each variable to have mean 0 and std 1
	meaned = feature_array - np.mean(feature_array, axis=0)
	feature_array = meaned / np.std(meaned, axis=0)

	np.savez("freddie_mac/np_arrays.npz", feature_array=feature_array, output_vector=output_vector)

	return feature_array, output_vector

if __name__ == "__main__":
	feature_array, output_vector = create_arrays()
