import pylab
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import numpy as np
import pandas as pd
import CBOW_DV as cdv


class ProbabilisticMatrixFactorization():
    def __init__(self, rating_tuples, latent_d=1):
        self.latent_d = latent_d
        self.learning_rate = .0001
        self.regularization_strength = 0.1

        ratings_temp = np.array(rating_tuples).astype(float)

        self.num_users = int(np.max(ratings_temp[:, 0]) + 1)
        self.num_items = int(np.max(ratings_temp[:, 1]) + 1)
        print('user numbers:', self.num_users)
        print('item numbers:', self.num_items)

        cut = int(len(ratings_temp) * 0.8)
        train_data = ratings_temp[0:cut]
        self.test_data = ratings_temp[cut:]
        self.ratings = np.array(train_data).astype(float)
        self.converged = False
        # print(self.num_users, self.num_items, self.latent_d)
        # print(self.ratings)

        self.users = np.random.random((self.num_users, self.latent_d))
        self.items = np.random.random((self.num_items, self.latent_d))
        self.initial_lik = -999999999

        self.new_users = np.random.random((self.num_users, self.latent_d))
        self.new_items = np.random.random((self.num_items, self.latent_d))

    def likelihood(self, users=None, items=None):
        if users is None:
            users = self.users
        if items is None:
            items = self.items

        sq_error = 0

        for rating_tuple in self.ratings:
            (i, j, rating) = rating_tuple
            i = int(i)
            j = int(j)
            r_hat = np.dot(users[i], items[j])
            sq_error += (rating - r_hat) ** 2

        l2_norm = 0
        for i in range(self.num_users):
            for d in range(self.latent_d):
                l2_norm += users[i, d] ** 2

        for i in range(self.num_items):
            for d in range(self.latent_d):
                l2_norm += items[i, d] ** 2

        return -sq_error - self.regularization_strength * l2_norm

    def update(self, times):
        updates_o = np.zeros(self.latent_d)
        updates_d = np.zeros(self.latent_d)

        # SGD
        ratings_len = len(self.ratings)
        rating_id = np.random.randint(0, ratings_len)
        rating_tuple = self.ratings[rating_id]

        (i, j, rating) = rating_tuple

        i = int(i)
        j = int(j)
        doc_key = str(j)
        # DV过程先更新
        cdv.DV_Enhanced_CBOW(doc_key)

        # 使用DV
        if doc_key not in cdv.docs.keys():
            r_hat = np.sum(self.users[i] * self.items[j])
        else:
            r_hat = np.sum(self.users[i] * (self.items[j] + cdv.doc_vec_table[cdv.doc_index[doc_key]]))

        # 记录 U，V的更新
        beta = -self.regularization_strength
        updates_o += self.items[j] * (rating - r_hat) + beta * self.users[i]
        updates_d += self.users[i] * (rating - r_hat) + beta * self.items[j]

        if times % 200 == 0:
            while not self.converged:
                print("  setting learning rate =", self.learning_rate)
                self.try_updates(updates_o, updates_d, i, j)

                items_with_dv = np.array(self.new_items)
                for z in range(len(items_with_dv)):
                    doc_key_temp = str(z + 1)
                    if doc_key_temp in cdv.docs.keys():
                        items_with_dv[z] += cdv.temp_update
                final_lik = self.likelihood(self.new_users, items_with_dv)

                if final_lik > self.initial_lik:
                    self.apply_updates(i, j)
                    self.learning_rate *= 1.25

                    if final_lik - self.initial_lik < .000001:
                        self.converged = True
                    self.initial_lik = final_lik
                    break

                else:
                    self.learning_rate *= .5
                    self.undo_updates()

                if self.learning_rate < 5e-5:
                    self.learning_rate = 5e-5
                    break
        else:
            self.direct_updates(updates_o, updates_d, i, j)

        return not self.converged

    def apply_updates(self, i, j):
        # self.new_users[i], self.new_items[j]  will change to other address next time
        for d in range(self.latent_d):
            self.users[i, d] = self.new_users[i, d]
            self.items[j, d] = self.new_items[j, d]
            # update dv
            # 反馈到DV
            doc_key = str(j)
            if doc_key in cdv.docs.keys():
                cdv.doc_vec_table[cdv.doc_index[doc_key]] = cdv.temp_update[d]

    def direct_updates(self, updates_o, updates_d, i, j):
        """
                try updates parameters
                :param updates_o:
                :param updates_d:
                :param i: user index
                :param j: item index
                :return:
                """
        alpha = self.learning_rate
        self.users[i] += alpha * updates_o
        self.items[j] += alpha * updates_d

        # update doc vector
        # 反馈到DV
        doc_key = str(j)
        if doc_key in cdv.docs.keys():
            cdv.doc_vec_table[cdv.doc_index[doc_key]] += alpha * updates_d

    def try_updates(self, updates_o, updates_d, i, j):
        """
        try updates parameters
        :param updates_o:
        :param updates_d:
        :param i: user index
        :param j: item index
        :return:
        """
        alpha = self.learning_rate
        beta = -self.regularization_strength

        self.new_users[i] = self.users[i] + alpha * (beta * self.users[i] + updates_o)
        self.new_items[j] = self.items[j] + alpha * (beta * self.items[j] + updates_d)

        # update doc vector
        # 反馈到DV
        doc_key = str(j)
        if doc_key in cdv.docs.keys():
            cdv.temp_update = cdv.doc_vec_table[cdv.doc_index[doc_key]] + alpha * updates_d

    def undo_updates(self):
        # Don't need to do anything here
        pass

    def print_latent_vectors(self):
        print("Users")
        for i in range(self.num_users):
            print(i),
            for d in range(self.latent_d):
                print(self.users[i, d]),

        print("Items")
        for i in range(self.num_items):
            print(i),
            for d in range(self.latent_d):
                print(self.items[i, d], )

    def save_latent_vectors(self, prefix):
        self.users.dump(prefix + "%sd_users.pickle" % self.latent_d)
        self.items.dump(prefix + "%sd_items.pickle" % self.latent_d)


def fake_ratings(noise=.25):
    u = []
    v = []
    ratings = []

    num_users = 100
    num_items = 100
    num_ratings = 30
    latent_dimension = 10

    # Generate the latent user and item vectors
    for i in range(num_users):
        u.append(2 * np.random.randn(latent_dimension))
    for i in range(num_items):
        v.append(2 * np.random.randn(latent_dimension))

    # Get num_ratings ratings per user.
    for i in range(num_users):
        items_rated = np.random.permutation(num_items)[:num_ratings]

        for jj in range(num_ratings):
            j = items_rated[jj]
            rating = np.sum(u[i] * v[j]) + noise * np.random.randn()

            ratings.append((i, j, rating))  # thanks sunquiang

    return ratings, u, v


def real_ratings():
    u = []
    v = []

    num_users = 100
    num_items = 100
    latent_dimension = 10

    # Generate the latent user and item vectors
    for i in range(num_users):
        u.append(2 * np.random.randn(latent_dimension))
    for i in range(num_items):
        v.append(2 * np.random.randn(latent_dimension))

    # Get ratings per user.
    movie_ratings = pd.read_csv(os.path.join('movielens', 'ratings.csv'))
    ratings = [tuple(x) for x in movie_ratings.values]
    result = [x[:-1] for x in ratings]
    return result, u, v


def plot_ratings(ratings):
    xs = []
    ys = []

    for i in range(len(ratings)):
        xs.append(ratings[i][1])
        ys.append(ratings[i][2])

    pylab.plot(xs, ys, 'bx')
    pylab.show()


def plot_latent_vectors(U, V):
    fig = plt.figure()
    ax = fig.add_subplot(121)
    cmap = cm.jet
    ax.imshow(U, cmap=cmap, interpolation='nearest')
    plt.title("Users")
    plt.axis("off")

    ax = fig.add_subplot(122)
    ax.imshow(V, cmap=cmap, interpolation='nearest')
    plt.title("Items")
    plt.axis("off")


def evaluate_predicted_ratings(ratings, U, V):
    r_hats = np.dot(U, V.transpose())
    rmse(ratings, r_hats)


def rmse(test_data, predicted):
    """Calculate root mean squared error.
    Ignoring missing values in the test data.
    """
    # I = ~np.isnan(test_data)   # indicator for missing values
    N = len(test_data)  # number of non-missing values
    sqerror = 0
    for tuple in test_data:
        sqerror += abs(tuple[2] - predicted[int(tuple[0])][int(tuple[1])]) ** 2
    mse = sqerror / N  # mean squared error
    print("RMSE: ", np.sqrt(mse))


if __name__ == "__main__":
    cdv.init()

    DATASET = 'real'
    if DATASET == 'fake':
        (ratings, true_o, true_d) = fake_ratings()
    if DATASET == 'real':
        (ratings, true_o, true_d) = real_ratings()

    #plot_ratings(ratings)

    print('rating numbers: ', len(ratings))
    pmf = ProbabilisticMatrixFactorization(ratings, latent_d=cdv.layer_size)
    pmf.initial_lik = pmf.likelihood()

    liks = []
    time = 0
    while pmf.update(time) and time <= 10000000:
        if time % 200 == 0:
            lik = pmf.likelihood()
            liks.append(lik)
            print("L=", lik, "; Time: ", time)
        if time % 2000 == 0:
            evaluate_predicted_ratings(pmf.test_data, pmf.users, pmf.items)
        time += 1
        pass

    evaluate_predicted_ratings(pmf.test_data, pmf.users, pmf.items)
    pmf.save_latent_vectors("models/")

    cdv.output()

    plt.figure()
    plt.plot(liks)
    plt.xlabel("Iteration")
    plt.ylabel("Log Likelihood")
    plot_latent_vectors(pmf.users, pmf.items)
    plt.show()
