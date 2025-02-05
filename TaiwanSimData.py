from TaiwanMechanism import TaiwanMechanism

class SimData:

    def __init__(self, num_instances, schools, students, rho=1, deduct=None, cap=None, comp_opt=True):
        self.num_instances = num_instances
        self.deduct = deduct
        self.rho = rho
        self.cap = cap

        self.instances = []
        self.generate_instances(schools, students, rho, deduct, cap, comp_opt)

    def generate_instances(self, schools, students, rho, deduct, cap, comp_opt):
        for i in range(self.num_instances):
            if i % 100 == 0:
                print("Generating instance " + str(i))
            self.instances.append(TaiwanMechanism(schools, students, rho, deduct, cap, comp_opt=comp_opt))

    def get_instance(self, i):
        return self.instances[i]

    def get_len(self):
        return self.num_instances

    def get_instances(self, i, j):
        return self.instances[i:j]

    def concat(self, other):
        self.instances += other.instances
        self.num_instances += other.num_instances
