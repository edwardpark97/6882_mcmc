import csv
import numpy as np

'''
Dependent Variable:
Whether they subscribed or not (index 20)

Predictor Variables:
Age in years (index 0) -> numerical age
Client's previous promotion outcome (index 14) -> "failure", "nonexistent", "success"
Type of contact (index 7) ->  "cellular", "telephone"
Education level (index 3) -> "basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate", "professional.course", "university.degree", "unknown"
Marital status (index 2) -> "divorced", "married", "single", "unknown"
'''

# Input parameter is a row from the csv file
# Returns a 18-length numpy vector
def create_row_features(r):
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

def create_arrays(data_path):
	if data_path == 'bank-additional/bank-additional.csv':
		num_data_points = 4119
	elif data_path == 'bank-additional/bank-additional-full.csv':
		num_data_points = 41188
	else:
		assert False
	num_features = 18 # number of columns from above 

	feature_array = np.zeros((num_data_points, num_features))
	output_vector = np.zeros(num_data_points)

	with open(data_path) as csvfile:
		reader = csv.reader(csvfile)
		next(reader) # skip the header row

		row_count = 0
		for row in reader:
			processed_row = row[0].replace("\"", "").split(";") # split the row into an array and get rid of quotation marks
			feature_array[row_count] = create_row_features(processed_row)
			output_vector[row_count] = int(processed_row[20] == "yes")
			row_count += 1

		# log and center the age feature
		age_vector = feature_array[:, 0]
		logged = np.log(age_vector)
		feature_array[:, 0] = logged - np.mean(logged)

		return feature_array, output_vector
			
if __name__ == "__main__":
	feature_array, output_vector = create_arrays()
	print(feature_array)
	print(output_vector)