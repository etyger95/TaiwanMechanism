import numpy as np
import random

def gen_random_prefs(r, rows, cols):
    """
    Generate preferences based on r (correlation parameter)
    :param r: correlation parameter, float between 0 and 1
    :param rows: number of rows
    :param cols: number of columns
    :return: numpy array of dim rows x cols
    """
    # r=0 means identical preferences:
    if r == 0:
        return np.tile(np.arange(cols), (rows, 1))
    # Otherwise, compute weights to use to randomly generate preference lists
    else:
        prefs = np.zeros((rows, cols))
        #  r=1 means uniform random weights
        if r == 1:
            weights = np.ones(cols) / cols
        else:
            weights = np.zeros(cols)
            rx = (r - 1) / (r ** cols - 1)
            for i in range(0, cols):
                weights[i] = rx * (r ** i)

        # generate priority lists using weights
        for i in range(0, rows):
            prefs[i] = np.random.choice(cols, cols, replace=False, p=weights)
        return prefs

def auto_readonly(cls):
    # Save the original __init__ method
    original_init = cls.__init__

    def new_init(self, *args, **kwargs):
        # Call the original __init__ to initialize instance attributes
        original_init(self, *args, **kwargs)

        # Add readonly properties for all `_`-prefixed attributes
        for attr in dir(self):
            if attr.startswith("_") and not attr.startswith("__"):
                prop_name = attr[1:]  # Remove leading underscore

                # Add the property dynamically
                setattr(
                    type(self),
                    prop_name,
                    property(lambda self, attr=attr: getattr(self, attr))
                )

    # Replace the class's __init__ with the new one
    cls.__init__ = new_init
    return cls

@auto_readonly
class TaiwanMechanism:
    def __init__(self, schools, students, rho=1.0, deduct=None, cap=None,
                 student_pref=None, student_rep=None,
                 school_pref_type='linear', school_pref_params=None,
                 comp_opt=False):
        """
        :param schools: int number of schools
        :param students: int number of students
        :param rho: float between 0 and 1, student preference correlation parameter
        :param deduct: array of length schools, deduction rule for each position in rank order list
        :param cap: int, capacity of each school (same for all schools)
        :param student_pref: numpy array of dim students x schools, student preferences over schools
        :param student_rep: numpy array of dim students x schools, student reported preferences over schools
        :param school_pref_type: string, type of school preferences.
            linear = score + w_g * geography + w_l * legacy
            random = generated like student preferences with correlation parameter
            manual = scores provided by user
        :param school_pref_params: depends on school_pref_type (default None)
            linear = numpy array of dim 2 x schools
            random = rho parameter (default 0, i.e., no correlation)
            manual = numpy array of dim students x schools
        :param comp_opt: boolean, whether to compute optimal reports and possibility arrays [generally not needed]
        """

        # Basic private attributes of the environment
        assert isinstance(schools, int) and schools > 0, "schools must be a positive integer"
        self._schools = schools
        assert isinstance(students, int) and students > 0, "students must be a positive integer"
        self._students = students
        assert 0 <= rho <= 1, "rho must be in [0,1]"
        self._rho = rho
        
        # if no rule specified, use Boston deduction rule
        self._deduct = deduct if deduct is not None else np.array([3 * i for i in range(self._schools)])
        assert isinstance(deduct, np.ndarray) and len(deduct) == self._schools, \
            "Deduction rule must be a numpy array of length schools"
        
        # if no cap specified, set to exactly fill each school
        self._cap = cap if cap is not None else int(students / schools) 
        assert isinstance(cap, int), "Cap must be an integer"

        # Preference parameters of the environment: private attributes of the environment
        self._gen_school_prefs(school_pref_type, school_pref_params)
        self._gen_student_prefs(rho, student_pref)

        # Compute student optimal stable match: useful to have this computed at initialization
        # Private because only depends on true preferences of the environment (not report)
        self._stu_opt = self.assign(truthful=True, da=True)



        # Reports can be changed: public attribute of the environment
        self.stu_report = self._stu_pref if student_rep is None else student_rep
        assert isinstance(self.stu_report, np.ndarray) and np.shape(self.stu_report) == (self._students, self._schools), \
            "Student reports must be a numpy array of shape students x schools"
        self.poss, self.optimal = None, None
        # Computing optimal and possible arrays is costly; can switch off
        if comp_opt:
            self.update_report(student_rep)

    def __str__(self):
        # If all entries of deduct are 3, Boston:
        if np.all(self._deduct == 3 * np.arange(self._schools)):
            return "Boston Mechanism with {} schools and {} students".format(self._schools, self._students)
        # If all entries of deduct are 0, DA:
        elif np.all(self._deduct == 0):
            return "Deferred Acceptance Mechanism with {} schools and {} students".format(self._schools, self._students)
        else:
            return "Taiwan Mechanism with {} schools and {} students, deduction rule {}".format(self._schools,
                                                                                                self._students, 
                                                                                                self._deduct)

    def _gen_school_prefs(self, pref_type, pref_params):
        """Generate school preferences as a numpy array of dim students x schools

        :param pref_type: string, type of school preferences; 'linear', 'random', or 'manual'
        # TODO: figure out how to do random so that there are scores for each student, not just an order of preference
        :param pref_params: parameters for generating school preferences; type depends on pref_type
        """
        if pref_type == 'linear':
            if pref_params is not None:
                assert pref_params.isinstance(dict), "School parameters must be a dictionary"

            w = np.random.rand(2, self._schools) if pref_params is None or 'w' not in pref_params \
                else pref_params['weights']
            assert w.shape == (2, self._schools), "Weights must be a numpy array of shape 2 x schools"

            scores = np.random.rand(self._students, ) if pref_params is None or 'scores' not in pref_params \
                else pref_params['scores']
            assert scores.shape == (self._students, ), \
                "Scores must be a numpy array of shape students"

            if pref_params is None or 'location' not in pref_params:
                # pick one neighborhood school for each student:
                location = np.zeros((self._students, self._schools))
                for i in range(self._students):
                    j = np.random.choice(self._schools)
                    location[i, j] = 1
            else:
                location = pref_params['location']
                assert location.shape == (self._students, self._schools), \
                    "Location must be a numpy array of shape students x schools"

            if pref_params is None or 'legacy' not in pref_params:
                # pick one "legacy" school for each student:
                legacy = np.zeros((self._students, self._schools))
                for i in range(self._students):
                    j = np.random.choice(self._schools)
                    legacy[i, j] = 1
            else:
                legacy = pref_params['legacy']
                assert legacy.shape == (self._students, self._schools), \
                    "Legacy must be a numpy array of shape students x schools"

            self._sch_pref = (scores[:, np.newaxis] + w[0] * location + w[1] * legacy).T

        # elif pref_type == 'random':
        #     if pref_params is None:
        #         pref_params = 0.0
        #     assert 0 <= pref_params <= 1, "School preference correlation param must be in [0,1]"
        #     self._sch_pref = gen_random_prefs(pref_params, self._schools, self._students)

        elif pref_type == 'manual':
            self._sch_pref = pref_params
            assert self._sch_pref.shape == (self._schools, self._students), \
                "School preferences must be a numpy array of shape schools x students"
        else:
            raise ValueError("School preference type must be 'linear', 'random', or 'manual'")

    def _gen_student_prefs(self, r, student_pref):
        """Generate preferences of students based on r (correlation parameter)

        :return: numpy array of dim students x schools
        """
        self._stu_pref = gen_random_prefs(r, self._students, self._schools) if student_pref is None else student_pref
        assert self._stu_pref.shape == (self._students, self._schools), \
            "Student preferences must be a numpy array of shape students x schools"

    def compute_rankings(self, rep_pref, da=False, return_dict=True, return_scores=False):
        """takes the reported preferences of students and deducts scores accordingly,
        returning the school's rank order list of students

        :param rep_pref: list of numpy arrays of dim schools x students of reported preferences
        :param da: option to run deferred acceptance (i.e., ignore deduction rule) on the rankings
        :param return_dict: option to return the rankings as a dictionary of {school: {student: rank}}
        :param return_scores: option to return the scores matrix instead of the rankings
        :return: numpy array of dim schools x students of rankings OR dictionary of rankings OR scores matrix
        """
        # initialize scores to very large negative number (i.e., score if never listed)
        scores_adj = np.ones((self._schools, self._students)) * (-99 * 10)

        for n in range(self._students):
            # Extract school indices from rep_pref
            curr_report = rep_pref[n]
            valid_indices = (curr_report != -1) & (curr_report < self._schools)
            unique_schools = np.unique(curr_report[valid_indices])
            # Calculate scores for listed schools
            for s in unique_schools:
                s = int(s)
                scores_adj[s, n] = self._sch_pref[s, n]
                if not da:
                    # Deduct from scores_adj based on reported position
                    scores_adj[s, n] -= self._deduct[np.where(curr_report == s)[0][0]]

        if return_scores:
            return scores_adj

        # Rank students based on adjusted scores
        rankings = np.argsort(-scores_adj, axis=1)
        if return_dict:
            rank_dict = [{} for _ in range(self._schools)]
            for s in range(self._schools):
                rank_dict[s] = {sidx: rank for rank, sidx in enumerate(rankings[s])}
            return rank_dict
        else:
            return rankings

    def assign(self, idx=None, report=None, return_all=True, truthful=False, da=False):
        """
        runs Taiwan Assignment Mechanism acceptance on the current instance
        :param idx: index of student to find match for [optional]
        :param report: report to use for student idx [optional]
        :param return_all: default to return all matches; if false will return only match for student idx
        :param truthful: option to use true preferences for all students
        :param da: option to run deferred acceptance (i.e., ignore deduction rule) on the reported rankings

        NOTE: assuming all students are acceptable for all schools, no one rejected if below capacity
        :return: numpy array of dim students x schools OR numpy array of dim students of school assignments for idx
        """
        # generate rankings:
        rep_pref = self._stu_pref.copy() if truthful else self.stu_report.copy()
        if not truthful and report is not None and idx is not None:
            assert len(report) == self._schools, "Report must be a list of length schools"
            rep_pref[idx] = report
        rep_pref = [np.array(pref) for pref in rep_pref]
        rankings = self.compute_rankings(rep_pref, da=da)

        # make sure no duplicates, delete -1 entries, add -1 entries to end of list
        for i in range(self._students):
            rep_pref[i] = rep_pref[i][rep_pref[i] != -1]
            seen = set()
            rep_pref[i] = [x for x in rep_pref[i] if x not in seen and not seen.add(x)]
            rep_pref[i] = np.append(rep_pref[i], -1)

        # create list for the SCHOOLS that students hold onto
        holds = np.full(self._students, -1)
        # create list for the proposals (STUDENTS) schools currently have
        proposals = {s: [] for s in range(self._schools)}
        # keep track of what INDEX the student has last proposed to
        curr_ind = np.zeros(self._students, dtype=int)

        # students propose to their first choice
        # if n has a school listed first (index zero), then propose to that school
        # and have n "hold" that school
        # (curr_ind[n] = 0 for all n, written like this to be consistent with future steps)
        for n in range(self._students):
            if int(rep_pref[n][curr_ind[n]]) != -1:
                # add n to their first choice school's proposal list
                proposals[int(rep_pref[n][curr_ind[n]])].append(n)
                # add n's first choice school to n's hold list
                holds[n] = int(rep_pref[n][curr_ind[n]])

        flag = True
        while flag:
            rejects = np.where(holds == -1)[0]
            # then, look at proposals to each school:
            for s in range(self._schools):
                # if more proposals than capacity, find rankings of all proposals and keep top ones
                if len(proposals[s]) > self._cap:
                    excess = len(proposals[s]) - self._cap
                    ranking = np.array([rankings[s][p] for p in proposals[s]])
                    excess_indices = np.argpartition(ranking, -excess)[-excess:]
                    excess_proposals = [proposals[s][i] for i in excess_indices]

                    rejects = np.append(rejects, excess_proposals)
                    proposals[s] = [p for p in proposals[s] if p not in excess_proposals]

            # if no one rejected from any school, then algorithm complete
            if len(rejects) == 0:
                flag = False

            # otherwise, have all rejects propose to next school
            else:
                # count will keep track of how many students can't propose to another school
                count = 0
                for n in rejects:
                    curr_ind[n] += 1
                    holds[n] = -1
                    if curr_ind[n] >= min(self._schools, len(rep_pref[n])):
                        count += 1
                    elif int(rep_pref[n][curr_ind[n]]) == -1:
                        count += 1
                    else:
                        proposals[int(rep_pref[n][curr_ind[n]])].append(n)
                        holds[n] = int(rep_pref[n][curr_ind[n]])

                # if no rejected students were able to propose to another school, we're done.
                if count == len(rejects):
                    flag = False

        # once algo is complete, return the index of the school agent idx is matched to (-1 if unmatched)
        if return_all:
            return holds
        return holds[idx]

    def assign_school_proposing(self, truthful=False, da=True):
        # Run DA with schools proposing to students

        # generate rankings:
        if truthful:
            rep_pref = self._stu_pref.copy()
        else:
            rep_pref = self.stu_report.copy()
        rep_pref = [np.array(pref) for pref in rep_pref]
        rankings = self.compute_rankings(rep_pref, da=da, return_dict=False)

        # make sure no duplicates, delete -1 entries
        # turn into dictionary for easy lookup
        for i in range(self._students):
            rep_pref[i] = rep_pref[i][rep_pref[i] != -1]
            seen = set()
            rep_pref[i] = [x for x in rep_pref[i] if x not in seen and not seen.add(x)]
            rep_pref[i] = {rep_pref[i][j]: j for j in range(len(rep_pref[i]))}

        # create list for the STUDENTS that schools hold onto
        holds = np.array([np.full(self._cap, -1) for _ in range(self._schools)])
        # create list for the proposals (SCHOOLS) students currently have
        proposals = {n: [] for n in range(self._students)}
        # keep track of what INDEX the school has last proposed to
        curr_ind = np.zeros(self._schools, dtype=int)

        # schools propose to their first choice
        # if s has a student listed first (index zero), then propose to that school
        # and have s "hold" that school
        # (curr_ind[s] = 0 for all s, written like this to be consistent with future steps)
        for s in range(self._schools):
            for c in range(self._cap):
                if int(rankings[s][curr_ind[s]]) != -1:
                    # add s to their first choice student's proposal list
                    proposals[int(rankings[s][curr_ind[s]])].append(s)
                    # add s's first choice student to s's hold list
                    holds[s][c] = int(rankings[s][curr_ind[s]])
                    if c != self._cap - 1:
                        curr_ind[s] += 1

        flag = True
        while flag:
            rejects = np.array([])
            num_rejects = np.zeros(self._schools, dtype=int)
            for s in range(self._schools):
                for c in range(self._cap):
                    if holds[s][c] == -1:
                        rejects = np.unique(np.append(rejects, s))
                        num_rejects[s] += 1

            # then, look at proposals to each student:
            for n in range(self._students):
                # if more proposals than capacity, find rankings of all proposals and keep top ones
                if len(proposals[n]) > 1:
                    # find favorite school in proposal list:
                    curr_rank = self._schools
                    curr_fav = None
                    for p in proposals[n]:
                        if p in rep_pref[n] and rep_pref[n][p] < curr_rank:
                            curr_fav = p
                            curr_rank = rep_pref[n][p]
                    if curr_fav is None:
                        excess_proposals = proposals[n]
                        proposals[n] = []
                    else:
                        excess_proposals = [p for p in proposals[n] if p != curr_fav]
                        proposals[n] = [curr_fav]

                    rejects = np.unique(np.append(rejects, excess_proposals))
                    for s in excess_proposals:
                        hold_pos = np.where(holds[s] == n)[0][0]
                        holds[s][hold_pos] = -1
                        num_rejects[s] += 1

                elif len(proposals[n]) == 1 and proposals[n][0] not in rep_pref[n]:
                    s = proposals[n][0]
                    hold_pos = np.where(holds[s] == n)[0][0]
                    holds[s][hold_pos] = -1
                    num_rejects[s] += 1
                    rejects = np.unique(np.append(rejects, s))
                    proposals[n] = []

            # if no school rejected, then algorithm complete
            if len(rejects) == 0:
                flag = False

            # otherwise, have all rejects propose to next student(s)
            else:
                # count will keep track of how many schools can't propose to another student
                count = 0
                for s in rejects:
                    s = int(s)
                    curr_ind[s] += 1
                    holds[s] = np.sort(holds[s])

                    for c in range(num_rejects[s]):
                        c = int(c)
                        if curr_ind[s] >= min(self._students, len(rankings[s])):
                            count += 1
                            break
                        elif int(rankings[s][curr_ind[s]]) == -1:
                            count += 1
                            break
                        else:
                            proposals[int(rankings[s][curr_ind[s]])].append(s)
                            holds[s][c] = int(rankings[s][curr_ind[s]])
                            if c != num_rejects[s] - 1:
                                curr_ind[s] += 1

                # if no rejected schools were able to propose to another student, we're done.
                if count == len(rejects):
                    flag = False

        assignment = np.full(self._students, -1)
        for n in range(self._students):
            for s in range(self._schools):
                if n in holds[s]:
                    assignment[n] = s
                    break

        return assignment

    def compute_cutoffs(self, real=True, sosm=False):
        """computes score cutoffs for each school based on rankings
        Score cutoffs are defined as the required score to be matched to a school
            if real is True, this is the raw score (not adjusted for deduction rule)
            if real is False, this is the adjusted score (adjusted for deduction rule)
        The SOSM parameter is used to generate the SOSM cut-offs (i.e., using the DA assignment)
        :return: numpy array of dim schools
        """
        cutoffs = np.zeros(self._schools)
        scores = self.compute_rankings(self.stu_report, da=real, return_scores=True)
        matches = self.assign(da=sosm, truthful=sosm)
        for s in range(self._schools):
            matched = np.where(matches == s)[0]
            if matched.size > 0:
                cutoffs[s] = np.min(scores[s, matched])
        return cutoffs

    def update_report(self, report=None):
        """
        Changes the student reports to the mechanism and updates the possible school and optimal school arrays
        This is used when computing "optimal" reports; not needed when not updating reports "optimally"
        :param report: report to update to; if none, uses the current true preferences
        """
        if report is None:
            self.stu_report = self._stu_pref.copy()
        else:
            assert isinstance(report, np.ndarray) and np.shape(report) == (self._students, self._schools), \
                "Report must be a numpy array of shape students x schools"
            # Reports are in terms of indices of true preferences, so convert to true preferences
            self.stu_report = np.where((report == -1) | (report >= self._schools), -1,
                                       self._stu_pref[np.arange(self._students)[:, None], report.astype(int)])
        # Update possible and optimal arrays based on new reports
        self.update_poss()
        self.update_optimal()

    def update_poss(self):
        """
        Updates the possible school array based on the current student reports.
        This is used when computing "optimal" reports; not needed when not updating reports "optimally"

        self.poss is a numpy array of dim students x schools where each entry is the
        index of the most desirable school achievable that student can get at that position
        in the rank order list
        """
        self.poss = -1 * np.ones((self._students, self._schools), dtype=int)
        for k in range(0, self._students):
            for i in range(0, self._schools):
                dev = np.full(self._schools, -1)
                pref_i = self._stu_pref[k][i]
                # start from the back and move up to find the highest index achievable
                for curr_ind in range(self._schools - 1, -1, -1):
                    # check the deviation where all -1 except putting i-th favorite at slot curr_ind
                    dev[curr_ind] = pref_i
                    x_check = self.assign(idx=k, report=dev, return_all=False)
                    dev[curr_ind] = -1  # Reset to -1 for next iteration

                    if x_check != -1:
                        assert self._stu_pref[k][i] == x_check, "Issue with achievable"
                        self.poss[k, i] = curr_ind
                        break

    def update_optimal(self):
        """
        Updates the optimal school array based on the current student reports.
        This is used when computing "optimal" reports; not needed when not updating reports "optimally"

        self.optimal is the optimal report the student should make:
        - lists favorite achievable school first,
        - then favorite achievable school when listing second, second
        - then favorite achievable school when listing third, third, etc.
        """
        self.optimal = -1 * np.ones((self._students, self._schools), dtype=int)
        for k in range(self._students):
            dev = np.full(self._schools, self._schools, dtype=int)
            poss = self.poss[k]
            curr_ind = 0
            # add all achievable schools in optimal order
            for i in range(self._schools):
                if poss[i] >= curr_ind:
                    dev[curr_ind] = i
                    curr_ind += 1
            self.optimal[k] = dev

    def smart_adjustment(self):
        """
        Adjusts student reports in a "smart" way:
        - if unassigned, reset to truthful
        - if you don’t get your first or second choice, drop first choice, shift everything up in order
        This adjustment is what results in Boston converging to SOSM and CPM converging to SOSM or better
        It does not necessarily converge to SOSM in the general Taiwan Mechanism
        """
        curr_assign = self.assign(truthful=False)
        for n in range(self._students):
            assigned = curr_assign[n]
            if np.where(self.stu_report[n] == assigned)[0][0] > 1:
                rep_pref = self.stu_report[n]
                rep_pref = rep_pref[1:]
                self.stu_report[n] = np.append(rep_pref, -1)

    def taiwan_smart_adjustment(self, rules_to_exclude=None, deterministic=None):
        """
        Adds additional rules to "smart" adjustment
        :param rules_to_exclude: list of rules to exclude (1, 2, or 3)
        :param deterministic: changes rule 3 to be deterministic (set to "first" or "last" to always select specific
        school preferred to assigned)
        May converge at matching that is Pareto dominated by SOSM, the SOSM, or a matching that Pareto dominates SOSM
        But does not always converge! Cycles are possible.
        Result is stored in self.stu_report
        """
        curr_assign = self.assign(truthful=False)

        for n in range(self._students):
            assigned = curr_assign[n]
            # Rule 1: if unassigned, reset to truthful
            if assigned == -1:
                if not (rules_to_exclude and 1 in rules_to_exclude):
                    self.stu_report[n] = self._stu_pref[n]

            # Rule 2: if you don’t get your first or second choice, drop first choice, shift everything up in order
            elif np.where(self.stu_report[n] == assigned)[0][0] > 1:
                if not (rules_to_exclude and 2 in rules_to_exclude):
                    self.stu_report[n] = np.append(self.stu_report[n][1:], -1)

            # Rule 3: if you’re getting your second choice, but there is a school you like better than your reported
            # first choice, swap out current first choice with a school you like better
            elif np.where(self.stu_report[n] == assigned)[0][0] == 1:
                if not (rules_to_exclude and 3 in rules_to_exclude):
                    index = np.where(self._stu_pref[n] == assigned)[0][0]
                    preferred_to_assigned = self._stu_pref[n][:index]
                    if index != 0:
                        if deterministic == "first":
                            # Select the first school in the list preferred to assigned
                            self.stu_report[n][0] = preferred_to_assigned[0]
                        elif deterministic == "last":
                            # Select the last school in the list preferred to assigned
                            self.stu_report[n][0] = preferred_to_assigned[-1]
                        else:
                            self.stu_report[n][0] = np.random.permutation(preferred_to_assigned)[0]

    def kmr_adjustment(self, p_drop, p_add, p_mutate, seed=18):
        """
        KMR adjustment to student reports
        Uses principles from Evolutionary Game Theory to adjust student reports
        :param p_drop: probability of dropping first choice
        :param p_add: probability of swapping top choice with a random school preferred to assigned
        :param p_mutate: probability of returning to true preferences [or randomly shuffling the list if desired]
        :param seed: to set random seed
        Result is stored in self.stu_report
        """
        random.seed(seed)
        curr_assign = self.assign(truthful=False)

        for n in range(self._students):
            assigned = curr_assign[n]
            # If unassigned, report true preferences [can change to random permutation if desired]
            if assigned == -1:
                # self.stu_report[n] = np.random.permutation(self._schools)
                self.stu_report[n] = self._stu_pref[n]

            else:
                # With probability p_drop, move first school listed (if not assigned there) to the end
                if random.random() < p_drop:
                    rep_pref = self.stu_report[n]
                    index = np.where(rep_pref == assigned)[0]
                    if index[0] != 0:
                        rep_pref = np.roll(rep_pref, -1)
                    self.stu_report[n] = rep_pref

                # With probability p_add, swap top choice with a random school preferred to assigned
                if random.random() < p_add:
                    # Create array of all schools preferred to assigned (or current top choice if unassigned)
                    index = np.where(self._stu_pref[n] == assigned)[0]
                    index = index[0]
                    preferred_to_assigned = self._stu_pref[n][:index]
                    # Select a random school from preferred_to_assigned
                    #  and swap this school with top choice if assigned not top choice
                    if index != 0:
                        preferred_to_assigned = np.random.permutation(preferred_to_assigned)[0]
                        if np.where(self.stu_report[n] == assigned)[0][0] != 0:
                            self.stu_report[n][0] = preferred_to_assigned
                        else:
                            rep_pref = self.stu_report[n]
                            rep_pref = np.concatenate((np.array([preferred_to_assigned]), rep_pref))[0:self._schools]
                            self.stu_report[n] = rep_pref

                # With probability p_mutate, report true preferences [can change to random permutation if desired]
                if random.random() < p_mutate:
                    # self.stu_report[n] = np.random.permutation(self._schools)
                    self.stu_report[n] = self._stu_pref[n]


