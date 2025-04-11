import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
from tqdm import tqdm
import matplotlib.pyplot as plt


class NestedCV_LinearRegressorWithEarlyStopping:
    def __init__(self, k_outer=5, k_inner=5, quantiles=[0.7, 0.8, 0.9, 0.95], early_stop_quantile=0.95, epsilon=0.00001, patience = 100):
        self._k_outer = k_outer  # Outer loop for evaluation
        self._k_inner = k_inner  # Inner loop for model selection
        self._model = LinearRegression()
        self._quantiles = quantiles
        self._all_errors = []  # Store outer errors
        self._test_errors = []
        self._inner_errors = []  # Store inner errors for each fold
        self._all_intervals = {}  # Store confidence intervals
        self._miscoverage_rates = {}  # Store miscoverage rates
        self._early_stop_quantile = early_stop_quantile
        self._epsilon = epsilon  # Threshold for early stopping
        self._patience = patience  # Number of iterations to wait before early stopping
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.all_test_mses = []
        self.previous_interval = None  # For early stopping
        self.total_fits = 0  # Track the number of fits done

    def run_on_data(self, X, y, n_repetitions=100):
        """
        Implements nested cross-validation following Algorithm 1 with early stopping.
        """
        max_fits = n_repetitions * self._k_outer * (1+self._k_inner) # Maximum number of outer fits
        early_stopped = False
        inner_loop_iterations = 0
        early_stopping_counter = 0
        stop_calc_intervals = False
        for _ in tqdm(range(n_repetitions), desc="Nested CV repetitions"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
            outer_kf = KFold(n_splits=self._k_outer, shuffle=True, random_state=None)

            for train_index, val_index in outer_kf.split(X_train):
                X_train_fold, X_val_fold = X[train_index], X[val_index]
                y_train_fold, y_val_fold = y[train_index], y[val_index]

                if stop_calc_intervals == False:
                    # Inner cross-validation to estimate the in-sample error
                    inner_errors = self.inner_crossval(X_train_fold, y_train_fold)
                    # Store inner error for later bias correction
                    self._inner_errors.append(np.mean(inner_errors))

                # Train the model on the full training data (excluding test set)
                    self._model.fit(X_train_fold, y_train_fold)
                    self.total_fits += 1  # Increment fit counter

                    # Evaluate the model on the outer test set
                    y_val_pred = self._model.predict(X_val_fold)
                    outer_error = mean_squared_error(y_val_fold, y_val_pred)
                    self._all_errors.append(outer_error)

                # Compute confidence intervals for the current fold
                if stop_calc_intervals == False:
                    confidence_intervals = self.compute_confidence_intervals(self._all_errors, self._inner_errors, self._early_stop_quantile)

                    # Check if the interval for the chosen quantile has stabilized (early stopping condition)
                    chosen_interval = confidence_intervals[self._early_stop_quantile]

                    if self.previous_interval is not None and inner_loop_iterations >= 1:
                        if (abs(chosen_interval[0] - self.previous_interval[0]) + abs(chosen_interval[1] - self.previous_interval[1])
                                < 2*self._epsilon):
                            #print(f"Early stopping triggered for quantile {self._early_stop_quantile} after {self.total_fits} fits.")

                            early_stopping_counter +=1
                            if early_stopping_counter > self._patience: #change to parameter patience
                                early_stopped = True
                                stop_calc_intervals = True
                    else:
                        early_stopping_counter = 0
                    # Update the previous interval for the next iteration
                    self.previous_interval = chosen_interval
                inner_loop_iterations += 1
                # if early_stopped:
                #     break  # Exit the repetition loop if early stopping is triggered

            # Calculate test set MSE
            y_test_pred = self._model.predict(X_test)
            test_mse = mean_squared_error(y_test, y_test_pred)
            self.all_test_mses.append(test_mse)

        self._all_intervals = self.compute_confidence_intervals(self._all_errors, self._inner_errors)
        self._miscoverage_rates = self._compute_miscoverage_rates()

        mean_error = np.mean(self._all_errors)

        # Print final statistics on early stopping
        print(f"Total fits performed: {self.total_fits} out of {max_fits} possible.")

        return mean_error, self._all_intervals, self._miscoverage_rates, self.total_fits, max_fits

    def inner_crossval(self, X, y):
        inner_kf = KFold(n_splits=self._k_inner, shuffle=True, random_state=None)
        inner_errors = []

        for inner_train_index, inner_val_index in inner_kf.split(X):
            X_inner_train, X_inner_val = X[inner_train_index], X[inner_val_index]
            y_inner_train, y_inner_val = y[inner_train_index], y[inner_val_index]

            # Train on inner training set
            self._model.fit(X_inner_train, y_inner_train)
            self.total_fits += 1  # Increment fit counter
            # Validate on the inner validation set
            y_inner_val_pred = self._model.predict(X_inner_val)
            inner_error = mean_squared_error(y_inner_val, y_inner_val_pred)

            inner_errors.append(inner_error)

        return inner_errors

    def compute_confidence_intervals(self, outer_errors, inner_errors, percentile_interval=None):
        """
        Computes confidence intervals for the provided quantiles.
        If `percentile_interval` is specified, only computes the confidence interval for that quantile.
        """
        K = len(outer_errors)  # Number of outer folds
        outer_errors = np.array(outer_errors)
        inner_errors = np.array(inner_errors)

        # Compute the mean error from outer folds
        mean_outer_error = np.mean(outer_errors)

        # Compute the bias term
        bias_term = (np.mean(inner_errors) - mean_outer_error) ** 2

        # Compute the MSE estimate
        mse_estimate = np.mean((outer_errors - mean_outer_error) ** 2)

        confidence_intervals = {}

        # If percentile_interval is provided, calculate only for that specific quantile
        if percentile_interval is not None:
            z_score = norm.ppf((1 + percentile_interval) / 2)
            ci_lower = mean_outer_error - bias_term - z_score * np.sqrt(mse_estimate)
            ci_upper = mean_outer_error - bias_term + z_score * np.sqrt(mse_estimate)
            confidence_intervals[percentile_interval] = (ci_lower, ci_upper)
        else:
            # Compute for all quantiles
            for level in self._quantiles:
                z_score = norm.ppf((1 + level) / 2)
                ci_lower = mean_outer_error - bias_term - z_score * np.sqrt(mse_estimate)
                ci_upper = mean_outer_error - bias_term + z_score * np.sqrt(mse_estimate)
                confidence_intervals[level] = (ci_lower, ci_upper)

        return confidence_intervals

    def _compute_miscoverage_rates(self):
        miscoverage_rates = {}

        for quantile in self._quantiles:
            lower_bound, upper_bound = self._all_intervals[quantile]
            outside_interval_count = sum(
                1 for error in self.all_test_mses if error < lower_bound or error > upper_bound
            )
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
        plt.title('Early stopping Confidence Intervals')

        # Move the legend to the right side of the graph
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
        plt.grid(True)
        plt.tight_layout()  # Adjust layout so the legend doesn't overlap
        plt.show()

def generate_linear_data(n_samples=1000, n_features=5, noise=0.2):
    X = np.random.randn(n_samples, n_features)
    true_coefficients = [ 0.19938878,  0.36693129, -0.83037629,  1.11561449, -1.22679938]
    y = X.dot(true_coefficients) + np.random.normal(loc=0, scale=1, size=n_samples) * noise
    return X, y, true_coefficients


class CvIntervalsWithEarlyStoppingTest:
    def __init__(self, n_repetitions=1000, quantiles=[0.7,0.8,0.9,0.95], k_outer=5, k_inner=5, epsilon=0.00001, patience=100):
        self.n_repetitions = n_repetitions
        self.quantiles = quantiles
        self.k_outer = k_outer
        self.k_inner = k_inner
        self.epsilon = epsilon
        self.patience = patience
    def run(self):
        X, y, _ = generate_linear_data(n_samples=10000, n_features=5, noise=0.34)
        regressor = NestedCV_LinearRegressorWithEarlyStopping(k_outer=5, k_inner=5, quantiles=self.quantiles,
                                                              epsilon  = self.epsilon, patience = self.patience)
        mean_error, intervals, miscoverage_rates, total_fits, max_fits = regressor.run_on_data(X, y, n_repetitions=self.n_repetitions)

        print(f"Estimated Prediction Error: {mean_error}")
        print(f"Confidence Intervals: {intervals}")
        print(f"Total Fits Performed: {total_fits} / {max_fits}")

        return mean_error, intervals, miscoverage_rates, total_fits, max_fits
        #regressor.plot_graph()

def main():
    test = CvIntervalsWithEarlyStoppingTest(n_repetitions=20, quantiles=[0.7, 0.8, 0.9, 0.95])
    test.run()


if __name__ == "__main__":
    main()
