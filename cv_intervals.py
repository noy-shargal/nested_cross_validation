import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm

class LinearRegressorWithClassicCV:
    def __init__(self, k, test_size=0.2, ci=[0.7, 0.8, 0.9, 0.95]):
        self._n_folds = k
        self._test_size = test_size
        self._model = LinearRegression()
        self._quantiles = ci

    def run_on_data(self, X, y, n_simulations=100):
        # Store all test MSE values for multiple simulations
        all_test_mses = []

        for _ in tqdm(range(n_simulations), desc="Simulations"):
            # Split the data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self._test_size, random_state=None)

            # Cross-validation on the training set
            kf = KFold(n_splits=self._n_folds, shuffle=True, random_state=None)
            errors = []
            for train_index, val_index in kf.split(X_train):
                X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
                y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

                # Train the model on the training fold
                self._model.fit(X_train_fold, y_train_fold)

                # Validate the model on the validation fold
                y_pred = self._model.predict(X_val_fold)
                fold_error = mean_squared_error(y_val_fold, y_pred)
                errors.append(fold_error)

            # Calculate MSE on the test set (final evaluation)
            y_test_pred = self._model.predict(X_test)
            test_mse = mean_squared_error(y_test, y_test_pred)
            all_test_mses.append(test_mse)

        # Compute confidence intervals for the cross-validation errors
        confidence_intervals = self.compute_confidence_intervals(errors, self._quantiles)

        return all_test_mses, confidence_intervals

    def compute_confidence_intervals(self, errors, confidence_levels):
        n = len(errors)
        errors = np.array(errors)

        # Calculate mean and standard error
        mean_error = np.mean(errors)
        std_error = np.std(errors, ddof=1) / np.sqrt(n)

        confidence_intervals = {}

        for level in confidence_levels:
            # Calculate the Z-score for the given confidence level using SciPy
            z_score = stats.norm.ppf((1 + level) / 2) #  1-(alpha/2) = (1+level)/2

            # Calculate the margin of error
            margin_of_error = z_score * std_error

            # Calculate the confidence interval
            ci_lower = mean_error - margin_of_error
            ci_upper = mean_error + margin_of_error

            confidence_intervals[level] = (ci_lower, ci_upper)

        return confidence_intervals


class CvIntervalsTest:

    def __init__(self, n_simulations=1000, quantiles=[0.7, 0.8, 0.9, 0.95],X_data=None, y_data=None):
        np.random.seed(42)  # For reproducibility
        self._n_simulations = n_simulations
        self._quantiles = quantiles
        # Arrays to store results
        self._all_errors = []
        self._all_intervals = []
        self._miscoverage_rates = {}
        self._X_data = X_data.to_numpy() if X_data is not None else None
        self._y_data = y_data.to_numpy() if y_data is not None else None

    def _compute_miscoverage_rates(self):
        """
        Computes the miscoverage rates for each quantile based on the errors and the corresponding quantile intervals.

        Miscoverage rate is calculated as the proportion of test samples where the true error falls outside
        the confidence interval for the corresponding quantile.
        """
        miscoverage_rates = {}

        total_samples = len(self._all_errors)  # Total number of test samples
        if total_samples == 0:
            raise ValueError("No test errors found.")

        # Loop over each quantile to calculate miscoverage rates
        for quantile in self._quantiles:
            print(f"\nEvaluating miscoverage rate for quantile: {quantile}")

            within_interval_count = 0  # Initialize a counter for samples within the interval

            # Retrieve the confidence interval for the current quantile
            lower_bound, upper_bound = self._all_intervals[quantile]

            # Loop through each sample's error
            for i, error in enumerate(self._all_errors):
                # Check if the error is within the bounds
                if lower_bound <= error <= upper_bound:
                    within_interval_count += 1

            # Calculate the miscoverage rate for this quantile
            miscoverage_rate = 1 - (within_interval_count / total_samples)

            print(f"Quantile {quantile}: Within Interval = {within_interval_count}, Total Samples = {total_samples}, "
                  f"Miscoverage Rate = {miscoverage_rate:.4f}")

            miscoverage_rates[quantile] = miscoverage_rate

        return miscoverage_rates

    def run(self):

        if self._X_data is None and self._y_data is None:
            # Generate data
            self._X_data, self._y_data, _ = generate_linear_data(n_samples=1000, n_features=5, noise=0.34)

        # Run regressor
        regressor = LinearRegressorWithClassicCV(k=5, test_size=0.2, ci=self._quantiles)
        test_mses, confidence_intervals = regressor.run_on_data(self._X_data,self._y_data, n_simulations=self._n_simulations)

        # Store all test errors and intervals
        self._all_errors = test_mses
        self._all_intervals = confidence_intervals

        print("\nConfidence Interval Sizes:")
        for quantile, (lower_bound, upper_bound) in confidence_intervals.items():
            interval_size = upper_bound - lower_bound
            print(f"Quantile: {quantile:.2f}, Interval Size: {interval_size:.4f}, "
                  f"Lower Bound: {lower_bound:.4f}, Upper Bound: {upper_bound:.4f}")

        # Calculate miscoverage rates
        self._miscoverage_rates = self._compute_miscoverage_rates()
        return self._miscoverage_rates
    def plot_graph(self):
        # Extract the quantiles and intervals
        quantiles = list(self._all_intervals.keys())
        lower_bounds = [self._all_intervals[q][0] for q in quantiles]
        upper_bounds = [self._all_intervals[q][1] for q in quantiles]

        plt.figure(figsize=(10, 6))

        # Plot confidence intervals as horizontal lines for each quantile
        for i, q in enumerate(quantiles):
            plt.hlines(lower_bounds[i], q - 0.02, q + 0.02, color='blue', linestyles='solid', label='Lower Bound' if i == 0 else "")
            plt.hlines(upper_bounds[i], q - 0.02, q + 0.02, color='red', linestyles='solid', label='Upper Bound' if i == 0 else "")

        # Plot all error points per quantile, using transparency (alpha) to show density
        for i, q in enumerate(quantiles):
            jittered_q = q + np.random.uniform(-0.01, 0.01, size=len(self._all_errors))  # Small jitter to avoid exact overlap
            plt.scatter(jittered_q, self._all_errors, color='green', s=5, alpha=0.4, label='Test Errors' if i == 0 else "")

        # Annotate miscoverage rates below each quantile
        for i, q in enumerate(quantiles):
            miscoverage_rate = self._miscoverage_rates[q]
            plt.text(q, lower_bounds[i] - 0.05, f'Miscoverage: {miscoverage_rate:.2f}', ha='center', va='top', fontsize=9, color='black')

        # Formatting the plot
        plt.xlabel('Quantiles')
        plt.ylabel('Error / Interval Bounds')
        plt.title('Test Errors and Confidence Intervals across Quantiles')

        # Move the legend to the upper right outside the plot
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
        plt.grid(True)
        plt.tight_layout()  # Adjust layout so the legend doesn't overlap

        plt.show()



def generate_linear_data(n_samples=1000, n_features=5, noise=0.2):
    X = np.random.randn(n_samples, n_features)
    true_coefficients = np.random.randn(n_features)
    y = X.dot(true_coefficients) + np.random.normal(loc=0, scale=1, size=n_samples) * 0.4
    return X, y, true_coefficients


def main():
    test = CvIntervalsTest(1000, [0.7, 0.8, 0.9, 0.95])
    test.run()
    test.plot_graph()


if __name__ == "__main__":
    main()
