import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
from tqdm import tqdm
import matplotlib.pyplot as plt


class LinearRegressorWithNestedCV:
    def __init__(self, k_outer=5, k_inner=5, quantiles=[0.7, 0.8, 0.9, 0.95]):
        self._k_outer = k_outer  # Outer loop for evaluation
        self._k_inner = k_inner  # Inner loop for model selection
        self._model = LinearRegression()
        self._quantiles = quantiles
        self._all_errors = []  # Store outer errors
        self._test_errors = []
        self._inner_errors = []  # Store inner errors for each fold
        self._all_intervals = {}  # Store confidence intervals
        self._miscoverage_rates = {}  # Store miscoverage rates

        self.X_train =None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.all_test_mses = []

    def run_on_data(self, X, y, n_repetitions=100):
        """
        Implements nested cross-validation following Algorithm 1.
        It estimates prediction error and MSE using the outer and inner cross-validation loops.
        """

        a_list = []  # List to store a terms
        b_list = []  # List to store b terms

        #all_test_mses = []
        # Repeat the nested cross-validation n_repetitions times
        for _ in tqdm(range(n_repetitions), desc="Nested CV repetitions"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
            outer_kf = KFold(n_splits=self._k_outer, shuffle=True, random_state=None)

            for train_index, val_index in outer_kf.split(X_train):
                X_train_fold, X_val_fold = X[train_index], X[val_index]
                y_train_fold, y_val_fold = y[train_index], y[val_index]

                # Inner cross-validation to estimate the in-sample error
                inner_errors = self.inner_crossval(X_train_fold, y_train_fold)

                # Store inner error for later bias correction
                self._inner_errors.append(np.mean(inner_errors))

                # Train the model on the full training data (excluding test set)
                model = LinearRegression()
                model.fit(X_train_fold, y_train_fold)
                #self._model.fit(X_train_fold, y_train_fold)

                # Evaluate the model on the outer test set
                y_val_pred = model.predict(X_val_fold)
                outer_error = mean_squared_error(y_val_fold, y_val_pred)

                # Store the outer test error
                self._all_errors.append(outer_error)
            self._model.fit(X_train, y_train)
            y_test_pred = self._model.predict(X_test)
            test_mse = mean_squared_error(y_test, y_test_pred)
            self.all_test_mses.append(test_mse)
            #
        # Compute confidence intervals based on the outer and inner errors
        self._all_intervals = self.compute_confidence_intervals(self._all_errors, self._inner_errors)

        # Evaluate the model on the global test set
        # glolbal_y_test_pred = self._model.predict(self.X_test)
        # self._test_errors = mean_squared_error(self.y_test, glolbal_y_test_pred)
        # Calculate MSE on the test set (final evaluation)


        # Compute the miscoverage rates after generating intervals
        self._miscoverage_rates = self._compute_miscoverage_rates()

        mean_error = np.mean(self._all_errors)
        return mean_error, self._all_intervals, self._miscoverage_rates

    def inner_crossval(self, X, y):
        """
        Inner cross-validation loop to estimate the in-sample error (e_in).
        """
        inner_kf = KFold(n_splits=self._k_inner, shuffle=True, random_state=None)
        inner_errors = []

        for inner_train_index, inner_val_index in inner_kf.split(X):
            X_inner_train, X_inner_val = X[inner_train_index], X[inner_val_index]
            y_inner_train, y_inner_val = y[inner_train_index], y[inner_val_index]

            # Train on inner training set
            self._model.fit(X_inner_train, y_inner_train)

            # Validate on the inner validation set
            y_inner_val_pred = self._model.predict(X_inner_val)
            inner_error = mean_squared_error(y_inner_val, y_inner_val_pred)

            inner_errors.append(inner_error)

        return inner_errors

    def compute_confidence_intervals(self, outer_errors, inner_errors):

        K = len(outer_errors)  # Number of outer folds
        outer_errors = np.array(outer_errors)
        inner_errors = np.array(inner_errors)

        # Compute the mean error from outer folds (Err_d(NCV) or \hat{\mu}_{NCV})
        mean_outer_error = np.mean(outer_errors)

        # Compute the bias term: (mean(inner_errors) - mean(outer_errors))^2
        bias_term = (np.mean(inner_errors) - mean_outer_error) ** 2

        # Compute the MSE estimate (plug-in estimator)
        mse_estimate = np.mean((outer_errors - mean_outer_error) ** 2)

        # Dictionary to hold the confidence intervals
        confidence_intervals = {}

        # Loop over the quantiles (confidence levels) provided in self._quantiles
        for level in self._quantiles:
            # Calculate the z-score for the given confidence level
            z_score = norm.ppf((1 + level) / 2)

            # Compute the lower and upper bounds for the confidence interval
            ci_lower = mean_outer_error - bias_term - z_score * np.sqrt(mse_estimate)
            ci_upper = mean_outer_error - bias_term + z_score * np.sqrt(mse_estimate)

            # Store the confidence interval for the current quantile
            confidence_intervals[level] = (ci_lower, ci_upper)

        return confidence_intervals

    # NEW: Method to compute the miscoverage rates
    def _compute_miscoverage_rates(self):
        """
        Computes the miscoverage rate for each quantile based on the errors and the corresponding confidence intervals.
        Miscoverage rate is calculated as the proportion of test samples where the true error falls outside
        the confidence interval for the corresponding quantile.
        """
        miscoverage_rates = {}

        # Loop over each quantile to calculate miscoverage rates
        for quantile in self._quantiles:
            lower_bound, upper_bound = self._all_intervals[quantile]

            # Count the number of errors outside the confidence interval
            outside_interval_count = sum(
                1 for error in self.all_test_mses if error < lower_bound or error > upper_bound
            )

            # Miscoverage rate = proportion of errors outside the interval
            miscoverage_rate = outside_interval_count / len(self.all_test_mses)

            miscoverage_rates[quantile] = miscoverage_rate

        return miscoverage_rates

    def plot_graph(self):
        # Extract the quantiles and intervals
        quantiles = list(self._all_intervals.keys())
        lower_bounds = [self._all_intervals[q][0] for q in quantiles]
        upper_bounds = [self._all_intervals[q][1] for q in quantiles]

        plt.figure(figsize=(10, 6))

        # Plot confidence intervals as horizontal lines for each quantile
        for i, q in enumerate(quantiles):
            plt.hlines(lower_bounds[i], q - 0.02, q + 0.02, color='blue', linestyles='solid',
                       label='Lower Bound' if i == 0 else "")
            plt.hlines(upper_bounds[i], q - 0.02, q + 0.02, color='red', linestyles='solid',
                       label='Upper Bound' if i == 0 else "")

        # Plot all error points per quantile, using transparency (alpha) to show density
        for i, q in enumerate(quantiles):
            jittered_q = np.repeat(q, len(self.all_test_mses)) + np.random.uniform(-0.01, 0.01,
                                                                                   size=len(self.all_test_mses))
            plt.scatter(jittered_q, self.all_test_mses, color='green', s=5, alpha=0.4,
                        label='Test Errors' if i == 0 else "")

        # Add miscoverage rates to the legend
        for i, q in enumerate(quantiles):
            miscoverage_rate = self._miscoverage_rates[q]
            plt.plot([], [], ' ', label=f'Quantile {q}: Miscoverage {miscoverage_rate:.2f}')

        # Formatting the plot
        plt.xlabel('Quantiles')
        plt.ylabel('Error / Interval Bounds')
        plt.title('Test Errors and Confidence Intervals across Quantiles')

        # Move the legend to the right side of the graph
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
        plt.grid(True)
        plt.tight_layout()  # Adjust layout so the legend doesn't overlap

        plt.show()


class CvIntervalsTest:
    def __init__(self, n_repetitions=100, quantiles =[0.7,0.8,0.9,0.95],k_outer=5, k_inner=5):
        self._n_repetitions = n_repetitions
        self._k_outer = k_outer
        self._k_inner = k_inner
        self._quantiles = quantiles

    def run(self):
        # Generate data
        X, y, _ = generate_linear_data(n_samples=10000, n_features=5, noise=0.34)

        # Run regressor with nested cross-validation
        regressor = LinearRegressorWithNestedCV(k_outer=self._k_outer, k_inner=self._k_inner, quantiles=self._quantiles)
        mean_error, intervals, miscoverage_rates = regressor.run_on_data(X, y, n_repetitions=self._n_repetitions)

        print(f"Estimated Prediction Error: {mean_error}")
        print("Confidence Intervals: ", intervals)

        #regressor.plot_graph()
        return mean_error, intervals, miscoverage_rates

def generate_linear_data(n_samples=1000, n_features=5, noise=0.2):
    """
    Generates linear data with Gaussian noise for the regression problem.
    """
    X = np.random.randn(n_samples, n_features)
    true_coefficients = [ 0.19938878,  0.36693129, -0.83037629,  1.11561449, -1.22679938] #np.random.randn(n_features)
    y = X.dot(true_coefficients) + np.random.normal(loc=0, scale=1, size=n_samples) * noise
    return X, y, true_coefficients


def main():
    test = CvIntervalsTest(n_repetitions=1000, k_outer=5, k_inner=5)
    test.run()


if __name__ == "__main__":
    main()
