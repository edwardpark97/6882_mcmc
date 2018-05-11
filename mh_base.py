import math
import numpy as np
import matplotlib.pyplot as plt
from pathos.multiprocessing import ProcessingPool as Pool
import process_bank
import process_freddie
import os
import time

PARALLEL = 1

class MCMCSampler(object):
	def __init__(self, log_f, log_g, g_sample, x0, iterations):
		"""
			Initialize a Metropolis-Hastings sampler for a target distribution P

			log_f: the log of a target distribution (or function proportional to the target distribution) that we will use to calculate acceptance ratios
			g: the log of a function that samples from the marginal probability density that defines the transition probabilities P(x|y) between samples, returns p(x|y) when given inputs x, y
			g_sample: a function that takes in a state x and returns a randomly sampled state in P(*|x)
			x0: the starting sample 
			iterations: number of iterations to ruh MH for
		"""
		self.log_distribution_fn = log_f
		self.log_transition_probabilities = log_g
		self.get_transition_sample = g_sample
		self.start_state = x0
		self.state = x0
		self.iterations = iterations

		self.saved_states = [x0]

	def sample(self):
		raise NotImplementedError

	def calculate_acceptance_ratio(self, proposal_state):
		raise NotImplementedError

	def transition_step(self, cur_state, candidate_state, acceptance_ratio):
		u = np.random.uniform()
		return cur_state if u > acceptance_ratio else candidate_state

	def get_saved_states(self):
		return self.saved_states

class MHSampler(MCMCSampler):
	def sample(self):
		for i in range(self.iterations):
			if i % 500 == 0:
				print("iteration {}".format(i))
			candidate_state = self.get_transition_sample(self.state)
			acceptance = self.calculate_acceptance_ratio(candidate_state)
			new_state = self.transition_step(self.state, candidate_state, acceptance)
			self.saved_states.append(new_state)
			self.state = new_state

	def calculate_acceptance_ratio(self, proposal_state):
		log = self.log_distribution_fn(proposal_state) + self.log_transition_probabilities(self.state, proposal_state) - self.log_distribution_fn(self.state) - self.log_transition_probabilities(proposal_state, self.state)
		if log > 0:
			acceptance_ratio = 1
		elif log < -20:
			acceptance_ratio = 0
		else:
			acceptance_ratio = np.exp(log)
		return min(1, acceptance_ratio)

class ConsensusMHSampler(MCMCSampler):
	def __init__(self, log_f, log_g, g_sample, x0, iterations, shards=1):
		super(ConsensusMHSampler, self).__init__(log_f, log_g, g_sample, x0, iterations)
		self.shards = shards

		assert len(self.log_distribution_fn) == self.shards
		self.log_fn_dict = {} # for pickling purposes
		for i in range(self.shards):
			self.log_fn_dict[i] = self.log_distribution_fn[i]

		self.pool = Pool(nodes=self.shards)

	def sample(self):
		map_results = self.pool.map(self.map_sample, range(self.shards))
		self.pool.close()
		self.pool.join()
		self.saved_states = self.reduce_sample(map_results)

	def map_sample(self, index):
		np.random.seed(1)
		cur_state = self.start_state
		sample_results = [cur_state]
		for i in range(self.iterations):
			if i % 500 == 0:
				print("iteration {}".format(i))
			candidate_state = self.get_transition_sample(cur_state)
			acceptance = self.calculate_acceptance_ratio(candidate_state, index)
			new_state = self.transition_step(cur_state, candidate_state, acceptance)
			sample_results.append(new_state)
			cur_state = new_state
		sample_results = np.array(sample_results)

		return (sample_results, 1.0 / (1e-8 + get_sample_variance(sample_results)))

	def reduce_sample(self, results):
		'''
			results is a list of (sample_array, weight) tuples
		'''
		sample_results = 0
		total_weight = 0
		for sample, weight in results:
			sample_results += weight * sample
			total_weight += weight

		return sample_results / total_weight

	def calculate_acceptance_ratio(self, proposal_state, index=0):
		log_fn = self.log_fn_dict[index]
		log = log_fn(proposal_state) + self.log_transition_probabilities(self.state, proposal_state) - log_fn(self.state) - self.log_transition_probabilities(proposal_state, self.state)
		if log > 0:
			acceptance_ratio = 1
		elif log < -20:
			acceptance_ratio = 0
		else:
			acceptance_ratio = np.exp(log)
		# print("%s %s" % (index, min(1, acceptance_ratio)))
		return min(1, acceptance_ratio)

class TwoStageMHSampler(MHSampler):
	def __init__(self, log_f1, log_f2, log_g, g_sample, x0, iterations):
		super(TwoStageMHSampler, self).__init__(log_f1, log_g, g_sample, x0, iterations)

		self.log_f1 = log_f1
		self.log_f2 = log_f2
		self.stage_1_prob, self.stage_1_count = 0, 0
		self.stage_2_prob, self.stage_2_count = 0, 0
		self.ratio_time, self.ratio_time_count = 0, 0

	def sample(self):
		for i in range(self.iterations):
			start_time = time.time()
			if i % 500 == 0:
				print("iteration {}".format(i))
			candidate_state = self.get_transition_sample(self.state)
			log_acceptance_ratio1 = self.calculate_log_acceptance_ratio1(candidate_state) # store for accep_ratio2 calculation
			acceptance1 = self.calculate_acceptance_ratio1(log_acceptance_ratio1)
			time1 = time.time() - start_time

			if np.random.uniform() < acceptance1: # do second stage
				acceptance2 = self.calculate_acceptance_ratio2(candidate_state, log_acceptance_ratio1)
				if np.random.uniform() < acceptance2:
					# print("moved")
					self.state = candidate_state
				self.ratio_time += (time.time() - start_time) / time1
				self.ratio_time_count += 1

			self.saved_states.append(self.state)

		print("Avg speedup per stage 2 skipped is {}".format(self.ratio_time / self.ratio_time_count))  
		print("Avg stage 1 prob is {}, avg stage 2 prob is {}".format(self.stage_1_prob/self.stage_1_count, self.stage_2_prob/self.stage_2_count))

	def calculate_log_acceptance_ratio1(self, proposal_state):
		return self.log_f1(proposal_state) + self.log_transition_probabilities(self.state, proposal_state) - self.log_f1(self.state) - self.log_transition_probabilities(proposal_state, self.state)

	def calculate_acceptance_ratio1(self, log):
		if log > 0:
			acceptance_ratio = 1
		elif log < -20:
			acceptance_ratio = 0
		else:
			acceptance_ratio = np.exp(log)
		self.stage_1_count += 1
		self.stage_1_prob += min(1, acceptance_ratio)
		return min(1, acceptance_ratio)

	def calculate_acceptance_ratio2(self, proposal_state, log_accept_ratio1):
		log = self.log_f2(proposal_state) - self.log_f2(self.state) - log_accept_ratio1
		if log > 0:
			acceptance_ratio = 1
		elif log < -20:
			acceptance_ratio = 0
		else:
			acceptance_ratio = np.exp(log)
		self.stage_2_count += 1
		self.stage_2_prob += min(1, acceptance_ratio)
		return min(1, acceptance_ratio)

def get_sample_variance(data):
	return np.linalg.norm(np.var(np.array(data), axis=0))

def calculate_ess(samples): # uses matplotlib to calculate ESS for each dimension, return min
	num_samples, num_features = samples.shape[0], samples.shape[1]
	zeroed = samples - np.mean(samples, axis=0)
	samples = zeroed / np.std(zeroed, axis=0)

	ess_by_dimension = np.zeros(num_features)
	for i in range(num_features):
		_, c, _, _ = plt.gca().acorr(samples[:, i], maxlags=num_samples-1)
		ess_by_dimension[i] = num_samples / (1 + 2 * sum(c[c.size//2:]))
	return np.min(ess_by_dimension)

def calculate_edpm(ess, seconds):
	return ess * 60. / seconds

# using main MH on the bank data
def generate_samples(dataset, sampling_method, a=400000):
	np.random.seed(1)
	if dataset == "bank_small":
		# N=100000 takes 1-2 min
		proposal_variance, N = .0015, 100000
		feature_array, output_vector = process_bank.create_arrays('bank-additional/bank-additional.csv')
		x0 = [-2.5, .1, -.2, -.35, -.05, -1, .5, .2, -.2, -.1]
	elif dataset == "bank_large":
		# N=20000 takes 1-2 min, N=4000000 takes ~5 hours
		proposal_variance, N = .00015, 20000
		feature_array, output_vector = process_bank.create_arrays('bank-additional/bank-additional-full.csv')
		x0 = [-2.45, .02, -.125, -.35, -.14, -.75, .35, .1, .1, -.45]
	elif dataset == "freddie_mac":
		# N=2000 takes about 7 min, N=100000 takes ~6 hours
		N = 5000
		if sampling_method == "MH":
			proposal_variance = 4e-5
		elif sampling_method == "consensus":
			proposal_variance = 8e-5
		elif sampling_method == "TwoStage":
			proposal_variance = 8e-6
		feature_array, output_vector = process_freddie.create_arrays()
		x0 = [-6.25, -.72, -.23, .56, .04, .13, .05, -.06, 0, .30]
	else:
		assert False

	print("\nUsing dataset {}, sampling_method {}, a={}, N={}, proposal_variance={}".format(dataset, sampling_method, a, N, proposal_variance))
	burnin = int(.4 * N)
	num_points, num_features = feature_array.shape[0], feature_array.shape[1]
	prior_mean, prior_variance = np.zeros(num_features), 100

	def log_multivariate_gaussian_pdf(x, y, variance):
		# assuming the covariance matrix is identity * variance
		return (-0.5 * (x - y).T.dot(np.identity(num_features) / variance).dot(x - y)) - (num_features / 2) * np.log(2 * np.pi * variance)

	def log_g_transition_prob(data, given):
		return log_multivariate_gaussian_pdf(data, given, proposal_variance)

	def g_sample(val):
		return np.random.multivariate_normal(val, np.identity(num_features) * proposal_variance)

	# used as a basis for all the log_f functions in the various sampling methods
	def f(val, features, outputs, shards=1):
		theta = features.dot(val)
		p_data = np.where(outputs, theta - np.log(1 + np.exp(theta)), - np.log(1 + np.exp(theta)))
		prior = log_multivariate_gaussian_pdf(prior_mean, val, prior_variance)
		return np.sum(p_data) + prior / shards # actually is a log value

	if sampling_method == "consensus":
		p = np.random.permutation(num_points)
		feature_array, output_vector = feature_array[p], output_vector[p]
		split_indices = []
		for i in range(1, 4):
			split_indices.append(int(math.floor(1./4 * i * num_points)))
		split_feature_array = np.split(feature_array, split_indices)
		split_output_vector = np.split(output_vector, split_indices)        

		log_fns = [lambda val: f(val, split_feature_array[0], split_output_vector[0], 4),
				lambda val: f(val, split_feature_array[1], split_output_vector[1], 4),
				lambda val: f(val, split_feature_array[2], split_output_vector[2], 4),
				lambda val: f(val, split_feature_array[3], split_output_vector[3], 4)]

		start_time = time.time()
		sampler = ConsensusMHSampler(log_fns, log_g_transition_prob, g_sample, x0, N, shards=4)
	
	elif sampling_method == "MH":
		log_f_distribution_fn = lambda val: f(val, feature_array, output_vector)
		start_time = time.time()
		sampler = MHSampler(log_f_distribution_fn, log_g_transition_prob, g_sample, x0, N)
	
	elif sampling_method == "TwoStage":
		log_f2 = lambda val: f(val, feature_array, output_vector)
		true_indices, false_indices = np.nonzero(output_vector)[0], np.nonzero(1 - output_vector)[0]

		# a = 400000 # number of false entries to subsample for the first stage (out of ~2million)
		num_false_entries = len(false_indices)
		false_indices = np.random.choice(false_indices, a, replace=False)
		true_feature_array, false_feature_array = feature_array[true_indices], feature_array[false_indices]

		def log_f1(val):
			# does calculations separately between true + false subsampled data points
			theta = true_feature_array.dot(val)
			true_contribution = np.sum(theta - np.log(1 + np.exp(theta)))
			theta = false_feature_array.dot(val)
			false_contribution = num_false_entries / a * np.sum(- np.log(1 + np.exp(theta)))
			prior = log_multivariate_gaussian_pdf(prior_mean, val, prior_variance)
			return true_contribution + false_contribution + prior
		start_time = time.time()
		sampler = TwoStageMHSampler(log_f1, log_f2, log_g_transition_prob, g_sample, x0, N)
	else:
		assert False

	sampler.sample()
	runtime = time.time() - start_time
	print("{} run time: {}\n".format(sampling_method, runtime))

	samples = sampler.get_saved_states()
	samples = np.array(samples[burnin:])

	np.save("samples/{}_{}_{}_{}_{}.npy".format(sampling_method, burnin, N, int(runtime), a), samples)
	return samples, runtime

def plot_tracking(samples, index, directory_path, sampling_method):
	plt.figure(1, figsize=(5, 10))

	plt.subplot(211)
	plt.title(r"PDF for Weight Parameter $\beta_{%s}$" % index)
	plt.hist(samples[:, index], 60, density=True)
	plt.xlabel("Value")
	plt.ylabel("Normalized Probability")

	plt.subplot(212)
	factor = int(math.ceil(len(samples) / 2000.)) # only plot 2000 points to make the plot look nicer
	subsamples = samples[::factor, index]
	plt.plot(subsamples, factor * np.arange(len(subsamples)), '.')
	plt.title(r"Distribution over Iterations of Weight Parameter $\beta_{%s}$" % index)
	plt.xlabel("Value")
	plt.ylabel("Sample Iteration")

	if not os.path.exists(directory_path):
		os.makedirs(directory_path)

	plt.savefig("%s/%s_%s.png" % (directory_path, index, sampling_method), bbox_inches='tight')
	plt.clf()

# a is the # of datapoints chosen for the first stage of twostage
def main(dataset, sampling_method, sample_path="", a=400000):
	if os.path.isfile(sample_path):
		samples = np.load(sample_path)

		file_name = sample_path.split("/")[-1].split(".")[0]
		runtime = float(file_name.split("_")[3]) # assuming same naming conventions samplingMethod_burnin_N_runtime.npy
	else:
		samples, runtime = generate_samples(dataset, sampling_method, a=a)

	for i in range(num_features):
	  plot_tracking(samples, i, "plots/{}".format(dataset), sampling_method)

	effective_sample_size = calculate_ess(samples)
	effective_draws_per_min = calculate_edpm(effective_sample_size, runtime)
	print("ESS: {}".format(effective_sample_size))
	print("EDPM: {}\n".format(effective_draws_per_min))

if __name__ == '__main__':
	# main("freddie_mac", "TwoStage", sample_path="samples/consensus_4000_10000_2736.npy")
	# main("freddie_mac", "TwoStage", sample_path="samples/MH_4000_10000_4262.npy")
	# main("freddie_mac", "TwoStage", sample_path="samples/TwoStage_4000_10000_2839.npy")
