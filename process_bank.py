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

DATA_PATH = "bank-additional/bank-additional-full.csv"
if "full" in DATA_PATH:
	NUM_DATA_POINTS = 41188
else:
	NUM_DATA_POINTS = 4119
NUM_FEATURES = 18 # number of columns from above 

# Input parameter is a row from the csv file
# Returns a 18-length numpy vector
def create_row_features(r):
	x = [r[0],
		r[14] == "failure",
		r[14] == "nonexistent",
		r[14] == "success",
		r[7] == "cellular",
		r[7] == "telephone",
		r[3] == "basic.4y",
		r[3] == "basic.6y",
		r[3] == "basic.9y",
		r[3] == "high.school",
		r[3] == "illiterate",
		r[3] == "professional.course",
		r[3] == "university.degree",
		r[3] == "unknown",
		r[2] == "divorced",
		r[2] == "married",
		r[2] == "single",
		r[2] == "unknown"]
	x = list(map(int, x))
	assert sum(x[1:]) == 4 # exactly four of these should be 1
	return np.array(x)

def create_arrays():
	feature_array = np.zeros((NUM_DATA_POINTS, NUM_FEATURES))
	output_vector = np.zeros(NUM_DATA_POINTS)

	with open(DATA_PATH) as csvfile:
		reader = csv.reader(csvfile)
		next(reader) # skip the header row

		row_count = 0
		for row in reader:
			processed_row = row[0].replace("\"", "").split(";") # split the row into an array and get rid of quotation marks
			feature_array[row_count] = create_row_features(processed_row)
			output_vector[row_count] = int(processed_row[20] == "yes")
			row_count += 1

		# center the age feature
		age_vector = feature_array[:, 0]
		feature_array[:, 0] = age_vector - np.mean(age_vector)

		return feature_array, output_vector
			
if __name__ == "__main__":
	feature_array, output_vector = create_arrays()
	print(feature_array)
	print(output_vector)