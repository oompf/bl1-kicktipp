import datetime, dateutil.parser
from scipy.optimize import curve_fit
import numpy as np
import math
import sys

# Globale Konfigurationsparameter
SKIP_GAMES = 306 * 5

class Match:
    def __init__(self, teams, start_time, season, result=None):
        self.teams = teams
        self.start_time = start_time
        self.season = season
        if result:
            self.is_finished = True

            if result[0] > result[1]:
                self.outcome = 1
            elif result[0] < result[1]:
                self.outcome = 0
            else:
                self.outcome = 0.5
        else:
            self.is_finished = False
            self.outcome = None
        self.result = result

    @staticmethod
    def from_json(j):
        # 1. Namen der Teams auslesen
        t1 = j["Team1"]["TeamName"]
        t2 = j["Team2"]["TeamName"]

        if t1 == "FC Bayern München":
            t1 = "FC Bayern"
        if t2 == "FC Bayern München":
            t2 = "FC Bayern"

        # 2. Zeitpunkt des Spiels auslesen
        start_time = dateutil.parser.parse(j["MatchDateTimeUTC"])
        season = int(j["LeagueName"][-9:-5])

        # 3. Spielergebnis auslesen
        g1 = -1
        g2 = -1
        for res in j["MatchResults"]:
            g1 = max(g1, res["PointsTeam1"])
            g2 = max(g2, res["PointsTeam2"])

        if min(g1, g2) < 0:
            return Match((t1, t2), start_time, season)
        else:
            return Match((t1, t2), start_time, season, (g1, g2))
    
    def goal_sum(self):
        return self.result[0] + self.result[1]

class ScalarRegression:
    def __init__(self):
        self.xs = []
        self.ys = []

    def add(self, x, y, max_vals=int(SKIP_GAMES/1.5)):
        self.xs.append(x)
        self.ys.append(y)

        if len(self.xs) >= max_vals:
            self.xs = self.xs[-max_vals:]
            self.ys = self.ys[-max_vals:]
    
    @staticmethod
    def f_func(x, a, b):
        return a * b**x

    def predict(self, x):
        res = curve_fit(ScalarRegression.f_func, self.xs, self.ys, method='dogbox', bounds=[0, np.inf])
        params = res[0]
        return ScalarRegression.f_func(x, *params)

class GlickoRating:
    # Konstanten
    Q = math.log(10) / 400
    C = 39.41
    TIME_DIV = (1 / 34) * datetime.timedelta(days=365)

    def __init__(self):
        self.last_game = {}
        self.rs = {}
        self.rds = {}

    def add_non_existing(self, match):
        for t in match.teams:
            if t not in self.rs:
                self.rs[t] = 1500
                self.rds[t] = 350
                self.last_game[t] = match.start_time
    
    def timeshift_rd(self, t, time):
        time_delta = (time - self.last_game[t]) / GlickoRating.TIME_DIV

        old_rd = self.rds[t]
        new_rd = math.sqrt(old_rd ** 2 + time_delta * GlickoRating.C ** 2)
        return min(350, new_rd)

    def expect(self, match, hfa):
        self.add_non_existing(match)
        
        t1, t2 = match.teams
        r1 = self.rs[t1]
        r2 = self.rs[t2]
        
        rd1 = self.timeshift_rd(t1, match.start_time)
        rd2 = self.timeshift_rd(t2, match.start_time)
        rd = math.sqrt(rd1**2 + rd2**2)

        return 1 / (1 + math.exp(GlickoRating.Q * (hfa + self.g_func(rd) * (r2 - r1))))

    def __train(self, t1, t2, outcome, time, hfa):
        rd1 = self.timeshift_rd(t1, time)
        rd2 = self.timeshift_rd(t2, time)

        r1 = self.rs[t1]
        r2 = self.rs[t2]

        g = self.g_func(rd2)

        e = 1 / (1 + math.exp(GlickoRating.Q * (hfa + g * (r2 - r1))))
        d = ((GlickoRating.Q * g) ** 2) * e * (1 - e)

        new_r1 = r1 + GlickoRating.Q / (rd1**-2 + d) * g * (outcome - e)
        new_rd1 = (rd1**-2 + d) ** (-0.5)

        self.rs[t1] = new_r1
        self.rds[t1] = new_rd1

    def update(self, match, hfa):
        self.add_non_existing(match)

        t1, t2 = match.teams

        outcome = match.outcome
        time = match.start_time
        self.__train(t1, t2, outcome, time, hfa)
        self.__train(t2, t1, 1 - outcome, time, -hfa)
        self.last_game[t1] = match.start_time
        self.last_game[t2] = match.start_time

    def g_func(self, rd):
        return 1 / math.sqrt(1 + 3 * (GlickoRating.Q * rd)**2 / math.pi**2)

    def __str__(self):
        lst = []
        for k in self.rds.keys():
            lst.append((self.rs[k], self.timeshift_rd(k, datetime.datetime.now(datetime.timezone.utc)), k))
        lst.sort(reverse=True)

        t = ""
        for l in lst:
            t = "{}\n{:10.3f} {:10.3f}   {}".format(t, l[0], l[1], l[2])
        return t

class MatchList:
    def __init__(self):
        self.matches = []
        self.avg_goals = []
        self.home_advs = []
        self.rating = GlickoRating()
        self.l1_predictor = ScalarRegression()
        self.l2_predictor = ScalarRegression()
        self.reward_matrices = [[] for _ in range(0, 5)]

    def append(self, match):
        self.matches.append(match)
    
    def compute(self):
        # 1. Liste der Spiele chronologisch sortieren
        self.matches.sort(key = lambda m : m.start_time)

        # 2. Durchschnittliche erwartete Torzahl und Heimvorteil pro Spiel bestimmen
        home_factors = []

        goal_sum = sum(map(lambda m : m.goal_sum(), self.matches[0:SKIP_GAMES]))  
        home_sum = sum(map(lambda m : m.outcome, self.matches[0:SKIP_GAMES]))

        for i in range(0, SKIP_GAMES):
            self.avg_goals.append(goal_sum / SKIP_GAMES)
            home_factors.append(home_sum / SKIP_GAMES)

        for i in range(SKIP_GAMES, len(self.matches)):
            if self.matches[i].is_finished:
                goal_sum -= self.matches[i-SKIP_GAMES].goal_sum()
                goal_sum += self.matches[i].goal_sum()
                home_sum -= self.matches[i-SKIP_GAMES].outcome
                home_sum += self.matches[i].outcome
                self.avg_goals.append(goal_sum / SKIP_GAMES)
                home_factors.append(home_sum / SKIP_GAMES)
            else:
                self.avg_goals.append(self.avg_goals[-1])
                home_factors.append(home_factors[-1])
        
        for h_factor in home_factors:
            self.home_advs.append(math.log(1 / h_factor - 1) / GlickoRating.Q)

        # 3. Matrizen berechnen
        self.calc_reward_matrices()

        # 4. Glicko-Modell trainieren
        bench = 0
        var = 0
        l1_diff = 0
        l2_diff = 0
        n = 0
        ee = 0
        for i in range(0, len(self.matches)):
            match = self.matches[i]

            if not match.is_finished:
                break

            # Benchmark
            if "--bench" in sys.argv:
                if match.season == 2019: #< 2019 and match.season >= 2014:
                    e1 = self.rating.expect(match, self.home_advs[i])
                    var += (e1 -  match.outcome)**2
                    n += 1
                    e2 = 1 - e1

                    ee += self.rating.rs[match.teams[0]] - self.rating.rs[match.teams[1]]

                    l1 = self.l1_predictor.predict(e1) * self.avg_goals[i]
                    l2 = self.l2_predictor.predict(e2) * self.avg_goals[i]

                    l1_diff += (match.result[0] - l1)
                    l2_diff += (match.result[1] - l2)

                    res = self.predict_kt(l1, l2)
                    tip1, tip2 = res[1]
                    bench += self.reward_matrices[tip1][tip2][match.result[0]][match.result[1]]

            # Lineare Regression trainieren
            if i >= SKIP_GAMES:
                avg_g = self.avg_goals[i]

                g1, g2 = match.result
                g1 = g1 / avg_g
                g2 = g2 / avg_g

                e1 = self.rating.expect(match, self.home_advs[i])
                e2 = 1 - e1
                
                self.l1_predictor.add(e1, g1)
                self.l2_predictor.add(e2, g2)

            self.rating.update(match, self.home_advs[i])
        if "--bench" in sys.argv:
            print("Benchmark: {}".format(bench))
            print("Glick-Predict: {:10.5f}".format(var / n))
            print("Goal-Predict: {:10.5f} {:10.5f}".format(l1_diff / n, l2_diff / n))
            print("r-diff: {:10.5f}".format(ee / n))
    
    def calc_reward_matrices(self):
        N = len(self.reward_matrices)
        for l in self.reward_matrices:
            for _ in range(0, N):
                l.append(None)
        
        for tip1 in range(0, N):
            for tip2 in range(0, N):
                # Unentschieden
                if tip1 == tip2:
                    reward = np.diag(2 * np.ones(11))
                else:
                    reward = np.zeros([11, 11])
                    for i1 in range(0, 11):
                        for i2 in range(0, 11):
                            if i1 - i2 == tip1 - tip2:
                                reward[i1][i2] = 3
                            elif ((i1 > i2) and (tip1 > tip2)) or ((i1 < i2) and (tip1 < tip2)):
                                reward[i1][i2] = 2
                reward[tip1][tip2] = 4
                
                self.reward_matrices[tip1][tip2] = reward

    def get_poisson_vector(self, lmbda):
        v = np.zeros(11)
        for k in range(0, len(v)):
            v[k] = math.pow(lmbda, k) / math.factorial(k) * math.exp(-lmbda)
        return np.asmatrix(v)

    def predict_kt(self, l1, l2):
        poi1 = self.get_poisson_vector(l1)
        poi2 = self.get_poisson_vector(l2)

        prob_matrix = poi1.transpose() * poi2

        N = len(self.reward_matrices)

        tips = []
        for tip1 in range(0, N):
            for tip2 in range(0, N):
                reward_matrix = self.reward_matrices[tip1][tip2]
                expected = np.sum(np.multiply(prob_matrix, reward_matrix))

                tips.append((expected, (tip1, tip2)))
        tips.sort(key = lambda x : x[0])       
        return tips[-1]

    def predict_upcoming(self):
        total_res = 0
        for i in range(0, len(self.matches)):
            match = self.matches[i]
            if match.is_finished:
                continue            

            e1 = self.rating.expect(match, self.home_advs[i])
            e2 = 1 - e1

            t1, t2 = match.teams

            l1 = self.l1_predictor.predict(e1) * self.avg_goals[i]
            l2 = self.l2_predictor.predict(e2) * self.avg_goals[i]

            res = self.predict_kt(l1, l2)
            total_res += res[0]

            print("{:>24}  ( {} : {} )  {:<24}   {:10.3f}".format(t1, res[1][0], res[1][1], t2, res[0]))
        print("\nTotal expected: {:10.4f}".format(total_res))
