import csv
import numpy as np

'''
Dependent Variable:
21 - y - has the client subscribed a term deposit? (binary: 'yes','no')

Predictor Variables (NOT DISCRETE):
Indexes are 1-indexed
1 - age (numeric)
12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14 - previous: number of contacts performed before this campaign and for this client (numeric)
16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
17 - cons.price.idx: consumer price index - monthly indicator (numeric) 
18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric) 
19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
20 - nr.employed: number of employees - quarterly indicator (numeric)

Predictor Variables (DISCRETE):
Indexes are 1-indexed
1 - age (numeric)
15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
8 - contact: contact communication type (categorical: 'cellular','telephone') 
4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
'''

def create_row_features(r):
	x = [1,			# 0 constant
		r[0],		# 1 age
		r[11],		# 2 campaign
		r[12],		# 3 pdays
		r[13],		# 4 previous
		r[15],		# 5 emp.var.rate
		r[16],		# 6 cons.price.idx
		r[17],		# 7 cons.conf.idx
		r[18],		# 8 euribor3m
		r[19]]		# 9 nr.employed
	return np.array(x)

# PROBABLY NOT USED ANYMORE
def create_row_features_discrete(r):
	x = [r[0],							# 0
		r[14] == "failure",				# 1
		r[14] == "nonexistent",			# 2
		r[14] == "success",				# 3
		r[7] == "cellular",				# 4
		r[7] == "telephone",			# 5
		r[3] == "basic.4y",				# 6
		r[3] == "basic.6y",				# 7
		r[3] == "basic.9y",				# 8
		r[3] == "high.school",			# 9
		r[3] == "illiterate",			# 10
		r[3] == "professional.course",	# 11
		r[3] == "university.degree",	# 12
		r[3] == "unknown",				# 13
		r[2] == "divorced",				# 14
		r[2] == "married",				# 15
		r[2] == "single",				# 16
		r[2] == "unknown"]				# 17
	x = list(map(int, x))
	assert sum(x[1:]) == 4 # exactly four of these should be 1
	return np.array(x)

def create_arrays(data_path, use_discrete_vars=False):
	if data_path == 'bank-additional/bank-additional.csv':
		num_data_points = 4119
	elif data_path == 'bank-additional/bank-additional-full.csv':
		num_data_points = 41188
	else:
		assert False
	num_features = 18 if use_discrete_vars else 10 # number of columns from above 

	feature_array = np.zeros((num_data_points, num_features))
	output_vector = np.zeros(num_data_points)

	with open(data_path) as csvfile:
		reader = csv.reader(csvfile)
		next(reader) # skip the header row

		row_count = 0
		for row in reader:
			processed_row = row[0].replace("\"", "").split(";") # split the row into an array and get rid of quotation marks
			feature_array[row_count] = create_row_features_discrete(processed_row) if use_discrete_vars \
										else create_row_features(processed_row)
			output_vector[row_count] = int(processed_row[20] == "yes")
			row_count += 1

		# log and center the age feature
		# age_vector = feature_array[:, 0]
		# logged = np.log(age_vector)
		# feature_array[:, 0] = logged - np.mean(logged)

		# norm each variable (except the constant term) to have mean 0 and std 1
		meaned = feature_array[:, 1:] - np.mean(feature_array[:, 1:], axis=0)
		feature_array[:, 1:] = meaned / np.std(meaned, axis=0)

		return feature_array, output_vector
			
if __name__ == "__main__":
	feature_array, output_vector = create_arrays('bank-additional/bank-additional-full.csv')
