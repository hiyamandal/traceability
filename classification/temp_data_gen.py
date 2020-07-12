cat = np.zeros(len(L))
for i in range(len(L)):
    alpha_0 = 1
    # factor = 0.00001
    factor = 0.01
    factor2 = 2

    p_blurring = [0.90, 0.10]
    # nachbearbeiten und ausschuss
    if L[i] > 2.5:
        if L[i] < 3:
            prior = np.asarray(np.array(p_blurring))
            cat[i] = np.random.multinomial(1, prior, size=1)[0][0]
        else:
            alpha = np.abs((L[i] - 3)) * factor
            prior = np.random.dirichlet((alpha_0, alpha), 1)[0]
            cat[i] = np.random.multinomial(1, prior, size=1)[0][0]
            if L[i] > 4.5:
                cat[i] += 1
        # cat[i] = 1

    if D[i] > 7.5:
        if D[i] < 8:
            prior = np.asarray(np.array(p_blurring))
            cat[i] = np.random.multinomial(1, prior, size=1)[0][0]
        else:
            alpha = np.abs((D[i] - 8)) * factor * factor2
            prior = np.random.dirichlet((alpha_0, alpha), 1)[0]
            cat[i] = np.random.multinomial(1, prior, size=1)[0][0]
            if D[i] > 9 or T[i] > 50 or T[i] < 13:
                cat[i] += 1
            # cat[i] = 1

    if D[i] < 4.5:
        if D[i] > 4:
            prior = np.asarray(np.array(p_blurring))
            cat[i] = np.random.multinomial(1, prior, size=1)[0][0]
        else:
            alpha = np.abs((D[i] - 4)) * factor * factor2
            prior = np.random.dirichlet((alpha_0, alpha), 1)[0]
            cat[i] = np.random.multinomial(1, prior, size=1)[0][0]
            if D[i] < 3 or T[i] < 13 or T[i] > 50:
                cat[i] += 1
            # cat[i] = 1

    if T[i] > 40:
        if T[i] < 45:
            prior = np.asarray(np.array(p_blurring))
            cat[i] = np.random.multinomial(1, prior, size=1)[0][0]
        else:
            alpha = np.abs((T[i] - 45)) * factor * factor2
            prior = np.random.dirichlet((alpha_0, alpha), 1)[0]
            cat[i] = np.random.multinomial(1, prior, size=1)[0][0]
            if T[i] > 50 or D[i] > 9:
                cat[i] += 1
            # cat[i] = 1

    if T[i] < 22:
        if T[i] > 18:
            prior = np.asarray(np.array(p_blurring))
            cat[i] = np.random.multinomial(1, prior, size=1)[0][0]
        else:
            alpha = np.abs((T[i] - 18)) * factor * factor2
            prior = np.random.dirichlet((alpha_0, alpha), 1)[0]
            cat[i] = np.random.multinomial(1, prior, size=1)[0][0]
            if T[i] < 13 or D[i] < 3:
                cat[i] += 1
            # cat[i] = 1

    if D[i] > 8.5 or D[i] < 3.5 or T[i] < 16 or T[i] > 50:
        prior = np.asarray(np.array([0.5, 0.5]))
        cat[i] = np.random.multinomial(1, prior, size=1)[0][0]
        cat[i] = cat[i] + 1

    if D[i] > 9 or D[i] < 3 or T[i] < 14 or T[i] > 52:
        prior = np.asarray(np.array([0.8, 0.2]))
        cat[i] = np.random.multinomial(1, prior, size=1)[0][0]
        cat[i] = cat[i] + 1

    if D[i] > 10 or D[i] < 2 or T[i] < 12 or T[i] > 54:
        prior = np.asarray(np.array([0.9, 0.1]))
        cat[i] = np.random.multinomial(1, prior, size=1)[0][0]
        cat[i] = cat[i] + 1

    if D[i] > 11 or D[i] < 1 or T[i] < 10 or T[i] > 56:
        cat[i] = 2
        # cat[i] = 2

    # Gutteil
    if D[i] > 5 and D[i] < 7 and T[i] > 18 and T[i] < 45 and L[i] > 2.5 and L[i] < 6:
        if T[i] > (7.71 * L[i] - 1.28):
            # prior1 = np.random.dirichlet((alpha_0, alpha), 1)[0]
            prior2 = np.asarray(np.array([0.95, 0.03, 0.02]))
            cat[i] = np.random.multinomial(1, prior2, size=1)[0][0]
        # if T[i] > (7.71 * L[i] - 1.28):
        #     cat[i] = 0