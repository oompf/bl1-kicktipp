import datetime, dateutil.parser
from scipy import stats
from scipy.optimize import minimize
import numpy as np
import math

# Globale Konfigurationsparameter
ELO_K = 23.5
SKIP_GAMES = 306 * 2
AUFSTEIGER_P = 0.2956 + 0.2587 / 2 # Wkt. für Sieg/Unterschieden für einen Aufsteiger

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

        # 3. Saison auslesen
        season = int(j["LeagueName"][-9:-5])

        # 4. Spielergebnis auslesen
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

class EloTable:
    def __init__(self):
        self.elos = {}

    def expect(self, teams):
        return 1 / (1 + 10**((self.elos[teams[1]] - self.elos[teams[0]]) / 500))

    def update(self, match):
        if not match.is_finished:
            return
        
        exp_oc = self.expect(match.teams)
        oc = match.outcome

        elo_delta = ELO_K * (oc - exp_oc)
                    
        self.elos[match.teams[0]] += elo_delta
        self.elos[match.teams[1]] -= elo_delta
    
    @staticmethod
    def optim_fct(elo, elo_array, n_aufsteiger):
        arr = 1 / (1 + 10 ** ((elo_array - elo) / 500))
        return abs((arr.sum() + (n_aufsteiger - 1) / 2) / (n_aufsteiger + len(arr) - 1) - AUFSTEIGER_P)
    
    def next_season(self, old_teams, new_teams):
        aufsteiger = new_teams - old_teams
        absteiger = old_teams - new_teams

        # Absteiger entfernen
        for k in absteiger:
            if k in self.elos:
                del self.elos[k]

        # Initiale Elo-Zahl für Aufsteiger bestimmen
        elo_list = []
        for k in self.elos.keys():
            elo_list.append(self.elos[k])
        elo_list = np.array(elo_list)
        n_aufsteiger = len(aufsteiger)

        res = minimize(EloTable.optim_fct, 250, (elo_list, n_aufsteiger))
        initial_elo = res.x[0]

        # Neue Teams initialisieren
        for k in aufsteiger:
            self.elos[k] = initial_elo
    
    def __str__(self):
        lst = []
        for k in self.elos:
            lst.append((self.elos[k], k))
        lst.sort(reverse=True)
        
        res = ""
        for l in lst:
            res = "{}{:10.4f}  {}\n".format(res, l[0], l[1])
        return res

class MatchList:
    def __init__(self):
        self.matches = []
        self.min_season = None
        self.max_season = None
        self.home_multi = None
        self.away_multi = None
        self.avg_goals = []
        self.teams_by_season = {}
        self.elo_table = EloTable()
        self.lin_regress = LinearRegression()
        self.reward_matrices = [[] for _ in range(0, 10)]

    def append(self, match):
        self.matches.append(match)
    
    def compute(self):
        # 1. Liste der Spiele chronologisch sortieren
        self.matches.sort(key = lambda m : m.start_time)

        self.min_season = self.matches[0].season
        self.max_season = self.matches[-1].season

        # 2. Heimvorteil bestimmen
        home_goals = 0
        away_goals = 0
        for match in self.matches:
            if match.is_finished:
                home_goals += match.result[0]
                away_goals += match.result[1]
        
        ha = home_goals / (home_goals + away_goals)

        self.home_multi = 2 * ha
        self.away_multi = 2 * (1 - ha)
        
        # 3. Teams nach Saison bestimmen
        for match in self.matches:
            if not match.season in self.teams_by_season:
                self.teams_by_season[match.season] = set()
            self.teams_by_season[match.season].add(match.teams[0])

        # 4. Durchschnittliche erwartete Torzahl pro Spiel bestimmen
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

        # 5. Matrizen berechnen
        self.calc_reward_matrices()

        # 6. Elo-Tabelle erstellen und Korrelation zur Torzahl herstellen
        self.elo_table.next_season(set(), self.teams_by_season[self.min_season])

        curr_season = self.min_season
        s = curr_season
        for i in range(0, len(self.matches)):
            match = self.matches[i]

            if not match.is_finished:
                break

            # Erkenne Wechsel der Saison
            s = match.season
            if s > curr_season:
                curr_season = s
                self.elo_table.next_season(self.teams_by_season[s - 1], self.teams_by_season[s])
            
            # Lineare Regression trainieren
            if i >= SKIP_GAMES:
                avg_goals = self.avg_goals[i]

                g1, g2 = match.result
                g1 = g1 / avg_goals
                g2 = g2 / avg_goals

                e1 = self.elo_table.expect(match.teams)
                e2 = 1 - e1
                
                self.lin_regress.add(e1, g1 / self.home_multi)
                self.lin_regress.add(e2, g2 / self.away_multi)

            self.elo_table.update(match)
    
    def calc_reward_matrices(self):
        N = len(self.reward_matrices)
        for l in self.reward_matrices:
            for _ in range(0, N):
                l.append(None)
        
        for tip1 in range(0, N):
            for tip2 in range(0, N):
                # Unentschieden
                if tip1 == tip2:
                    reward = np.diag(2 * np.ones(20))
                else:
                    reward = np.zeros([20, 20])
                    for i1 in range(0, 20):
                        for i2 in range(0, 20):
                            if i1 - i2 == tip1 - tip2:
                                reward[i1][i2] = 3
                            elif ((i1 > i2) and (tip1 > tip2)) or ((i1 < i2) and (tip1 < tip2)):
                                reward[i1][i2] = 2
                reward[tip1][tip2] = 4
                
                self.reward_matrices[tip1][tip2] = reward

    def get_poisson_vector(self, lmbda):
        v = np.zeros(20)
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
            if cnt >= 18:
                break

            cnt += 1

            e1 = self.elo_table.expect(match.teams)
            e2 = 1 - e1

            t1, t2 = match.teams

            l1 = self.lin_regress.predict(e1) * self.home_multi * self.avg_goals[i]
            l2 = self.lin_regress.predict(e2) * self.away_multi * self.avg_goals[i]

            res = self.predict_kt(l1, l2)
            total_res += res[0]

            print("{:>24}  ( {} : {} )  {:<24}   {:10.3f}".format(t1, res[1][0], res[1][1], t2, res[0]))
        print("\nTotal expected: {:10.4f}".format(total_res))