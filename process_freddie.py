import csv
import numpy as np

# Try downloading the numpy file from AWS
try:
	import boto3
	import botocore
	ACCESS_ID = 'ACCESS_ID_HERE'
	ACCESS_KEY = 'SECRET_KEY_HERE'
	AWS_BUCKET = '6882-mcmc'
	FREDDIE_MAC_FILEPATH = 'freddie_mac_arrays.npz'
	s3 = boto3.resource('s3', 
		aws_access_key_id=ACCESS_ID,
		aws_secret_access_key=ACCESS_KEY)
	s3.Bucket(AWS_BUCKET).download_file(FREDDIE_MAC_FILEPATH, 'freddie_mac/np_arrays.npz')
except Exception:
	pass

'''	
Dependent Variable:
whether a loan was foreclosed by the end of september 2017
	go into time-historical data,
	if column 9 is "03" then was foreclosed 

Predictor Variables:
Date of first mortgage payment (column 2)
FICO score (column 1)
Debt to income ratio (column 10)
Original principal balance of the loan (column 11)
First-time home-buyer status (column 3) - "yes", "no", "unknown"
'''

# Input parameter is a row from the csv file
# Returns a 7-length numpy vector
def create_row_features(r):
	def YYYYMM_to_month(date):
		return 12 * (int(date[:4]) - 2009) + int(date[4:])
	x = [YYYYMM_to_month(r[1]),
		r[0],
		r[9],
		r[10],
		r[2] == 'Y',
		r[2] == 'N',
		r[2] != 'Y' and r[2] != 'N']
	x = list(map(int, x))
	return np.array(x)

def create_arrays():
	# Try loading the numpy array
	try:
		np_array = np.load("freddie_mac/np_arrays.npz")
		return np_array['feature_array'], np_array['output_vector']
	except Exception as e:
		print(e)
		pass # if it doesn't work, recompute it

	q1, q2, q3, q4 = 587250, 654170, 381904, 350521 # num data points in each file
	num_data_points = q1 + q2 + q3 + q4
	num_features = 7 # number of columns from above 

	feature_array = np.zeros((num_data_points, num_features))
	output_vector = np.zeros(num_data_points)
	index_to_loan_sequence = np.zeros(num_data_points) # maps the numpy index to the loan sequence number

	# generate feature_array
	row_count = 0
	for i in range(1, 5):
		with open("freddie_mac/historical_data1_Q%s2009.txt" % i) as csvfile:
			reader = csv.reader(csvfile, delimiter="|")
			for row in reader:
				feature_array[row_count] = create_row_features(row)
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
	print(feature_array[:50])
	print(sum(output_vector))