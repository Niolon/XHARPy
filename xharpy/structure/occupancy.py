from dataclasses import dataclass
from .common import Value, FixedValue, RefinedValue, AtomicProperty

@dataclass(frozen=True)
class Occupancy(AtomicProperty):
    occ_value: Value
    special_position: bool
    derived: bool = False

    def resolve(self, parameters):
        return self.occ_value.resolve(parameters)

    def resolve_esd(self, var_cov_mat):
        return self.occ_value.resolve_esd(var_cov_mat)

    def occupancy(self, parameters):
        if self.special_position:
            if isinstance(self.occ_value, RefinedValue):
                return self.resolve(parameters) / self.occ_value.multiplicator
            elif isinstance(self.occ_value, FixedValue):
                return 1
            else:
                raise NotImplementedError('Cannot handle this parameters type')
        else:
            return self.occ_value.resolve(parameters)

    def occupancy_esd(self, var_cov_mat):
        if self.special_position and isinstance(self.occ_value, RefinedValue):
            return self.resolve_esd(var_cov_mat) / self.occ_value.multiplicator
        else:
            return self.occ_value.resolve_esd(var_cov_mat)

    def symmetry_order(self):
        if self.special_position: 
            if isinstance(self.occ_value, RefinedValue):
                return int(1 / self.occ_value.multiplicator)
            elif isinstance(self.occ_value, FixedValue):
                return int(1 / self.occ_value.value)
            else:
                raise NotImplementedError('Currently cannot handle this type of variable')
        else:
            return 1
        
