import numpy as np
import matplotlib.pyplot as plt
from process_bank import create_arrays

class MHSampler(object):
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
		self.state = x0
		self.iterations = iterations

		self.saved_states = [x0]

		self.step_count = 0

	def sample(self):
		for i in range(self.iterations):
			candidate_state = self.get_transition_sample(self.state)
			acceptance = self.calculate_acceptance_ratio(candidate_state)
			new_state = self.transition_step(candidate_state, acceptance)
			self.saved_states.append(new_state)
			self.state = new_state

			self.step_count += 1

	def calculate_acceptance_ratio(self, proposal_state):
		log = self.log_distribution_fn(proposal_state) + self.log_transition_probabilities(self.state, proposal_state) - self.log_distribution_fn(self.state) - self.log_transition_probabilities(proposal_state, self.state)
		if log > 0:
			acceptance_ratio = 1
		elif log < -20:
			acceptance_ratio = 0
		else:
			acceptance_ratio = np.exp(log)
		# print(min(1, acceptance_ratio))
		return min(1, acceptance_ratio)

	def transition_step(self, candidate_state, acceptance_ratio):
		u = np.random.uniform()
		return self.state if u > acceptance_ratio else candidate_state

	def get_saved_states(self):
		return self.saved_states

def main():
	# estimate mean given sig = 1
	true_mu, true_sig, N = 0, 1, 100000

	x0 = np.random.uniform(-1.0, 1.0)

	def log_f_distribution_fn(val):
		return np.log(np.exp(-val**2 / 2)/np.sqrt(2 * np.pi))

	def log_g_transition_prob(data, given):
		# assume we know sig = 1
		return np.log(np.exp(-(data - given)**2 / 2)/np.sqrt(2 * np.pi))

	def g_sample(val):
		return np.random.normal(loc = val, scale = true_sig)

	sampler = MHSampler(log_f_distribution_fn, log_g_transition_prob, g_sample, x0, N)
	sampler.sample()

	samples = sampler.get_saved_states()

	count, bins, ignored = plt.hist(np.array(samples), 30, normed=True)

	plt.plot(bins, 1/(true_sig * np.sqrt(2 * np.pi)) * np.exp( - (bins - true_mu)**2 / (2 * true_sig**2) ), linewidth=2, color='r')
	plt.show()

# using main MH on the bank data
def bank_main():
	proposal_variance, N = .0005, 100000 # TO BE DETERMINED such that acceptance prob is ~ 20%
	# use .0005 for the smaller bank dataset, .00005 for the larger bank dataset

	feature_array, output_vector = create_arrays('bank-additional/bank-additional.csv')
	num_features = feature_array.shape[1]
	prior_mean, prior_variance = np.zeros(num_features), 100

	x0 = np.random.multivariate_normal(prior_mean, np.identity(num_features) * prior_variance)
	x0 = [0.014, 2.5, 2.3, 5, -2.3, -3.2, -3, -3, -3, -3, 1, -3, -3, -3, .8, .8, 1, .8]
	# hard code this in rn to avoid burn-in

	def log_multivariate_gaussian_pdf(x, y, variance):
		# assuming the covariance matrix is identity * variance
		return (-0.5 * (x - y).T.dot(np.identity(num_features) / variance).dot(x - y)) - (num_features / 2) * np.log(2 * np.pi * variance)

	def log_f_distribution_fn(val):
		# calculated based on the formula in section 3.1 of the paper
		theta = feature_array.dot(val)
		p_data = np.where(output_vector, theta - np.log(1 + np.exp(theta)), - np.log(1 + np.exp(theta)))
		prior = log_multivariate_gaussian_pdf(prior_mean, val, prior_variance)
		return np.sum(p_data) + prior

	def log_g_transition_prob(data, given):
		return log_multivariate_gaussian_pdf(data, given, proposal_variance)

	def g_sample(val):
		return np.random.multivariate_normal(val, np.identity(num_features) * proposal_variance)

	sampler = MHSampler(log_f_distribution_fn, log_g_transition_prob, g_sample, x0, N)
	sampler.sample()

	samples = sampler.get_saved_states()
	samples = np.array(samples[20000:]) # burn-in time

	for i in range(num_features):
		plot_histogram(samples, i)


def plot_histogram(samples, index):
	count, bins, ignored = plt.hist(samples[:, index], 30, density=True)
	plt.xlabel("Value")
	plt.ylabel("Noramlized Probability")
	plt.title(r"PDF of Weight Parameter $\beta_{%s}$" % index)
	plt.savefig("plots/MH_bank/%s.png" % index, bbox_inches='tight')
	plt.clf()

if __name__ == '__main__':
	bank_main()
