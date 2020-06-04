import datetime, dateutil.parser
from scipy import stats
from scipy.optimize import minimize
import numpy as np
import math
import sys

# Globale Konfigurationsparameter
SKIP_GAMES = 306 * 2
HFA = 125 * math.log(10) / 400

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

class LinearRegression:
    def __init__(self):
        self.xs = []
        self.ys = []

    def add(self, x, y):
        self.xs.append(x)
        self.ys.append(y)
    
    def predict(self, x):
        slope, intercept, _, _, _ = stats.linregress(self.xs, self.ys)
        return intercept + slope * x

class GlickoRating:
    # Konstanten
    Q = math.log(10) / 400
    C = 30.41
    TAU = 1.2
    TIME_DIV = (1 / 34) * datetime.timedelta(days=365)

    def __init__(self):
        self.last_game = {}
        self.rs = {}
        self.rds = {}
        self.vols = {}

    def add_non_existing(self, match):
        for t in match.teams:
            if t not in self.rs:
                self.rs[t] = 1500
                self.rds[t] = 350
                self.vols[t] = GlickoRating.C * GlickoRating.Q
                self.last_game[t] = match.start_time
    
    def timeshift_rd(self, t, time):
        time_delta = (time - self.last_game[t]) / GlickoRating.TIME_DIV

        rd = self.rds[t]
        phi = rd * GlickoRating.Q

        new_phi = math.sqrt(phi ** 2 + time_delta * self.vols[t] ** 2)
        new_rd = min(350, new_phi / GlickoRating.Q)

        return new_rd

    def expect(self, match):
        self.add_non_existing(match)
        
        t1, t2 = match.teams
        r1 = self.rs[t1]
        r2 = self.rs[t2]
        
        rd1 = self.timeshift_rd(t1, match.start_time)
        rd2 = self.timeshift_rd(t2, match.start_time)
        rd = math.sqrt(rd1**2 + rd2**2)

        mu1 = (r1 - 1500) * GlickoRating.Q
        mu2 = (r2 - 1500) * GlickoRating.Q
        phi = rd * GlickoRating.Q

        return 1 / (1 + math.exp(self.g_func(phi) * (mu2 - (mu1 + HFA))))

    def __train(self, t1, t2, outcome, time, is_t1_home=True):
        time_delta = (time - self.last_game[t1]) / GlickoRating.TIME_DIV

        if is_t1_home:
            hfa_boost = HFA
        else:
            hfa_boost = -HFA

        time_delta = (time - self.last_game[t1]) / GlickoRating.TIME_DIV

        # Schritt 1
        rd1 = self.timeshift_rd(t1, time)
        rd2 = self.timeshift_rd(t2, time)

        r1 = self.rs[t1]
        r2 = self.rs[t2]

        mu1 = (r1 - 1500) * GlickoRating.Q
        mu2 = (r2 - 1500) * GlickoRating.Q

        phi1 = rd1 * GlickoRating.Q
        phi2 = rd2 * GlickoRating.Q

        vol1 = self.vols[t1]

        # Schritt 2

        # Schritt 2
        e = 1 / (1 + math.exp(self.g_func(phi2) * (mu2 - (mu1 + hfa_boost))))
        v = 1 / (self.g_func(phi2)**2 * e * (1 - e))

        delta = v * self.g_func(phi2) * (outcome - e)

        # Schritt 3
        res = minimize(self.f_func, vol1, (delta, phi1, v, vol1)) #, tol=1e-6)

        new_vol1 = res.x[0]

        new_phi1 = math.sqrt(phi1**2 + new_vol1**2)
        new_phi1 = (new_phi1**(-2) + 1/v)**(-1/2)

        new_mu1 = mu1 + new_phi1**2 * self.g_func(phi2) * (outcome - e)

        new_r1 = new_mu1 / GlickoRating.Q + 1500
        new_rd1 = new_phi1 / GlickoRating.Q

        self.rs[t1] = new_r1
        self.rds[t1] = new_rd1
        self.vols[t1] = new_vol1

    def update(self, match):
        self.add_non_existing(match)

        t1, t2 = match.teams
        outcome = match.outcome
        time = match.start_time
        self.__train(t1, t2, outcome, time, is_t1_home=True)
        self.__train(t2, t1, 1 - outcome, time, is_t1_home=False)
        self.last_game[t1] = match.start_time
        self.last_game[t2] = match.start_time

    def g_func(self, phi):
        return 1 / math.sqrt(1 + 3 * phi**2 / math.pi**2)

    def f_func(self, x, delta, phi, v, sigma):
        z = x**2 * (delta**2 - phi**2 - v - x**2) / (2 * (phi**2 + v + x**2)**2)
        z = z - math.log(x**2 / sigma**2) / GlickoRating.TAU ** 2
        return abs(z)

    def __str__(self):
        lst = []
        for k in self.rds.keys():
            lst.append((self.rs[k], self.timeshift_rd(k, datetime.datetime.now(datetime.timezone.utc)), self.vols[k], k))
        lst.sort(reverse=True)

        t = ""
        for l in lst:
            t = "{}\n{:10.3f} {:10.3f} {:10.3f}  {}".format(t, l[0], l[1], l[2], l[3])
        return t

class MatchList:
    def __init__(self):
        self.matches = []
        self.avg_goals = []
        self.rating = GlickoRating()
        self.home_mult = None
        self.away_mult = None
        self.lin_regress = LinearRegression()
        self.reward_matrices = [[] for _ in range(0, 10)]

    def append(self, match):
        self.matches.append(match)
    
    def compute(self):
        # 1. Liste der Spiele chronologisch sortieren
        self.matches.sort(key = lambda m : m.start_time)

        # 2. Durchschnittliche erwartete Torzahl pro Spiel bestimmen
        goal_sum = sum(map(lambda m : m.goal_sum(), self.matches[0:SKIP_GAMES]))  

        for i in range(0, SKIP_GAMES):
            self.avg_goals.append(goal_sum / SKIP_GAMES)

        for i in range(SKIP_GAMES, len(self.matches)):
            if self.matches[i].is_finished:
                goal_sum -= self.matches[i-SKIP_GAMES].goal_sum()
                goal_sum += self.matches[i].goal_sum()
                self.avg_goals.append(goal_sum / SKIP_GAMES)
            else:
                self.avg_goals.append(self.avg_goals[-1])

        # 4. Matrizen berechnen
        self.calc_reward_matrices()

        # 5. Glicko-Modell trainieren
        bench = 0
        for i in range(0, len(self.matches)):
            match = self.matches[i]

            if not match.is_finished:
                break

            # Benchmark
            if "--bench" in sys.argv:
                if match.season < 2019 and match.season >= 2014:
                    e1 = self.rating.expect(match)
                    e2 = 1 - e1

                    l1 = self.lin_regress.predict(e1) * self.avg_goals[i]
                    l2 = self.lin_regress.predict(e2) * self.avg_goals[i]

                    res = self.predict_kt(l1, l2)
                    tip1, tip2 = res[1]
                    bench += self.reward_matrices[tip1][tip2][match.result[0]][match.result[1]]

            # Lineare Regression trainieren
            if i >= SKIP_GAMES:
                avg_goals = self.avg_goals[i]

                g1, g2 = match.result
                g1 = g1 / avg_goals
                g2 = g2 / avg_goals

                e1 = self.rating.expect(match)
                e2 = 1 - e1
                
                self.lin_regress.add(e1, g1)
                self.lin_regress.add(e2, g2)

            self.rating.update(match)
        if "--bench" in sys.argv:
            print("Benchmark: {}".format(bench))
            
    
    def calc_reward_matrices(self):
        N = len(self.reward_matrices)
        for l in self.reward_matrices:
            for _ in range(0, N):
                l.append(None)
        
        for tip1 in range(0, N):
            for tip2 in range(0, N):
                # Unentschieden
                if tip1 == tip2:
                    reward = np.diag(2 * np.ones(25))
                else:
                    reward = np.zeros([25, 25])
                    for i1 in range(0, 25):
                        for i2 in range(0, 25):
                            if i1 - i2 == tip1 - tip2:
                                reward[i1][i2] = 3
                            elif ((i1 > i2) and (tip1 > tip2)) or ((i1 < i2) and (tip1 < tip2)):
                                reward[i1][i2] = 2
                reward[tip1][tip2] = 4
                
                self.reward_matrices[tip1][tip2] = reward

    def get_poisson_vector(self, lmbda):
        v = np.zeros(25)
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
        cnt = 0
        total_res = 0
        for i in range(0, len(self.matches)):
            match = self.matches[i]
            if match.is_finished:
                continue

        
            if cnt >= 14:
                break
            cnt += 1
            

            e1 = self.rating.expect(match)
            e2 = 1 - e1

            t1, t2 = match.teams

            l1 = self.lin_regress.predict(e1) * self.avg_goals[i]
            l2 = self.lin_regress.predict(e2) * self.avg_goals[i]

            res = self.predict_kt(l1, l2)
            total_res += res[0]

            print("{:>24}  ( {} : {} )  {:<24}   {:10.3f}".format(t1, res[1][0], res[1][1], t2, res[0]))
        print("\nTotal expected: {:10.4f}".format(total_res))
