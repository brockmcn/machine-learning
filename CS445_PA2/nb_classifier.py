"""Pure Python Naive Bayes classifier

Simple nb_classifier.

Initial Author: Kevin Molloy and Anthony Thellaeche and Brockton McNerney
"""


import numpy as np
import math
import scipy
from scipy import stats


class NBClassifier:
    """
    A naive bayes classifier for use with categorical and real-valued attributes/features.

    Attributes:
        classes (list): The set of integer classes this tree can classify.
        smoothing_flag (boolean): Indicator whether or not to perform
                                  Laplace smoothing
        feature_dists (list):  A placeholder for each feature/column in X
                               that holds the distributions for that feature.
    """

    def __init__(self, smoothing_flag=False):
        """
        NBClassifier constructor.

        :param smoothing: for discrete elements only
        """

        if smoothing_flag:
            self.smoothing = 1
        else:
            self.smoothing = 0

        """
        feature_dists is a list of dictionaries, one for each feature in X.
        The first dictionary is envisioned to be for each class label, and
        then the second level dictionaries is either:
          -- for continuous features, a tuple with the distribution
          parameters for a Gaussian (mu, std)
          -- for discrete features, another dictionary where the keys 
             are the individual domain values for the feature
             and the value is the computed probability from the training data 
        """
        self.feature_dists = []

    def get_smoothing(self):
        return self.smoothing

    def fit(self, X, X_categorical, y):
        """
        Construct the NB using the provided data and labels.

        :param X: Numpy array with shape (num_samples, num_features).
                  This is the training data.
        :param X_categorical: numpy boolean array with length num_features.
                              True values indicate that the feature is discrete.
                              False values indicate that the feature is continuous.
        :param y: Numpy integer array with length num_samples
                  These are the training labels.

        :return: Stores results in class variables, nothing returned.

        An example of how my dictionary looked after running fit on the
        loan classification problem in the textbook without smoothing:
        [{0: {'No': 0.5714285714285714, 'Yes': 0.42857142857142855},
        1: {'No': 1.0} },
        {0: {'Divorced': 0.14285714285714285, 'Married': 0.5714285714285714, 'Single': 0.2857142857142857},
         1: {'Divorced': 0.3333333333333333, 'Single': 0.6666666666666666}   },
        {0: (110.0, 54.543560573178574, 2975.0000000000005),
         1: (90.0, 5.0, 25.0)}]
        """

        ## Need a category for each column in X
        assert(X.shape[1] == X_categorical.shape[0])

        ## each row in training data needs a label
        assert(X.shape[0] == y.shape[0])

        ## additional self values
        self.classes = list(set(y))
        self.X_categorical = X_categorical
        self.X = X
        self.y = y
        self.priors = {}

        #For each feature (column)
        for feature_index in range(X.shape[1]):
            #add a dict for features to the list
            self.feature_dists.append({})

            # Get each possible value
            x_values = np.unique(X[:, feature_index])

            #ignore features that only have 1 value
            if(len(x_values) < 2): continue

            # For each label
            for class_label in range(len(self.classes)):

                #add a dict for values for each feature
                self.feature_dists[feature_index][class_label] = {}

                #Calculate each value for feature of categorical features
                if(self.X_categorical[feature_index]):

                    # For each possible value
                    for x in x_values:
                        self.feature_class_prob(feature_index, class_label, x)
                #Only need to do once for each class label for continuous features
                else:
                    self.feature_class_prob(feature_index, class_label, 0)

    def feature_class_prob(self, feature_index, class_label, x):
        """
        Compute a single conditional probability.  You can call
        this function in your predict function if you wish.

        Example: For the loan default problem:
            feature_class_prob(1, 0, 'Single') returns 0.5714

        :param feature_index: index into the feature set (column of X)
        :param class_label: the label used in the probability (see return below)
        :param x: the data value

        :return: P(class_label | feature(fi) = x) the probability
        """

        #current feature
        feature_dist = self.feature_dists[feature_index]

        # validate feature_index
        assert feature_index < self.X_categorical.shape[0], \
            'Invalid feature index passed to feature_class_prob'

        # validate class_label
        assert class_label < len(self.classes), \
            'invalid class label passed to feature_class_prob'

        # for categorical features
        if (self.X_categorical[feature_index]):

            # calculate probability
            ys = np.count_nonzero(self.y == class_label)
            xs = np.count_nonzero(np.logical_and(self.X[:, feature_index] == x, self.y == class_label))

            # smoothing
            if self.smoothing:
                v = len(np.unique(self.X[:, feature_index]))
                p_given = (xs + 1) / (ys + v)
            else:
                p_given = xs / ys

            #add p_given to feature_dists if not 0
            if(p_given != 0):
                self.feature_dists[feature_index][class_label][x] = p_given

        #for continuous features
        else:
            #get x_given
            indexes = np.where(self.y == class_label)
            x_given = self.X[:, feature_index][indexes[0]]

            # std deviation and mean
            std = np.std(x_given.astype(np.float), ddof=1)
            mean = np.mean(x_given.astype(np.float))

            # converts to float if necessary
            if isinstance(x, np.str_): xfloat = x.astype(float)
            else: xfloat = x

            #calculate probability
            p_given = scipy.stats.norm(mean,std).pdf(xfloat)

            #smoothing (set p_given to 10^-9 if 0)
            if(p_given == 0):
                p_given = 10 ** -9

            #add p_given to feature_dists
            self.feature_dists[feature_index][class_label] = (mean, std, std ** 2)

        return p_given

    def predict(self, X):
        """
        Predict labels for test matrix X

        Parameters/returns
        ----------
        :param X:  Numpy array with shape (num_samples, num_features)
        :return: Numpy array with shape (num_samples, )
            Predicted labels for each entry/row in X.
        """

        ## validate that x contains exactly the number of features
        assert(X.shape[1] == self.X_categorical.shape[0])

        preds = np.array([])

        #For each row
        for row in range(X.shape[0]):

            max_sum = 0
            pred = 0

            #For each class
            for cl in range(len(self.classes)):

                val = 1

                #For each feature
                for fi in range(len(self.feature_dists)):

                    #Get from list if categorical
                    if (self.X_categorical[fi]):

                        # Return 0 if not in list
                        if X[:,fi][row] in self.feature_dists[fi][cl]:
                            val *= self.feature_dists[fi][cl][X[:,fi][row]]
                        else:
                            val *= 0

                    #Calculate if continuous
                    else:
                        val *= self.feature_class_prob(fi, cl, X[:, fi][row])

                #exp val total
                if val > 0:
                    val = math.exp(math.log(val))

                #Set prediction if greater than previous
                if val > max_sum:
                    pred = cl
                    max_sum = val

            #Add prediction to array
            preds = np.append(preds, pred)

        return preds.astype(int)

def nb_demo():
    ## data from table Figure 4.8 in the textbook
    X = np.array([['Yes',   'Single',   125],   #NO
                  ['No',    'Married',  100],   #NO
                  ['No',    'Single',   70],    #NO
                  ['Yes',   'Married',  120],   #NO
                  ['No',    'Divorced', 95],    #YES
                  ['No',    'Married',  60],    #NO
                  ['Yes',   'Divorced', 220],   #NO
                  ['No',    'Single',   85],    #YES
                  ['No',    'Married',  75],    #NO
                  ['No',    'Single',   90]     #YES
                 ])

    ## first two features are categorical and 3rd is continuous
    X_categorical = np.array([True, True, False])

    ## class labels (default borrower)
    y = np.array([0, 0, 0, 0, 1, 0, 0, 1, 0, 1])

    nb = NBClassifier(smoothing_flag=False)

    nb.fit(X, X_categorical, y)

    test_pt = X
    yhat = nb.predict(test_pt)

    # the book computes this as 0.0016 * alpha
    print('Predicted value for someone who does not a homeowner,')
    print('is married, and earns 120K a year is:', yhat)


def main():
    nb_demo()

if __name__ == "__main__":
    main()

