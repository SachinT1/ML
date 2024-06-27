import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    datamodel = LogisticRegression(eps=1e-5)
    datamodel.fit(x_train, y_train)
    util.plot(x_train, y_train, model.theta, 'output/p01b_{}.png'.format(pred_path[-5]))
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval)
    np.savetxt(pred_path, y_pred > 0.5, fmt='%d')
   


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***

        def htheta(theta,x):
            return (1/(1+np.exp(-np.dot(x.theta))))
        
        def grad(theta,x,y):
            m = x.shape[0]
            return((-1/m)*(np.dot(x.T,)))

        def hess(theta,x):
            m = x.shape[0]
            het = np.reshape(htheta(theta,x),(-1,1))
            return ((1/m)*(np.dot(x.T,het*(1-het)*x)))

        def upd(theta,x,y):
            return theta - np.dot(np.linalg.inv(hess(theta,x)),grad(theta,x,y))
        
        m,n= x.shape

        if(self.theta ==None):
            self.theta = np.zeros(n)
        oldt = self.theta
        newt= upd(oldt,x,y)
        while np.linalg.norm(newt-oldt,1)>=self.eps:
            oldt = newt
            newt = upd(oldt,x,y)
        self.theta = newt


        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """

        return x@self.theta >=0
        # *** START CODE HERE ***
        # *** END CODE HERE ***
