from .discovery.graph import CausalDiscovery
from .estimation.estimators import CausalEstimation
from .auditor import CausalAuditor

class CausalBranch:
    def __init__(self, parent):
        self.parent = parent
        self.discovery = CausalDiscovery(self)
        self.estimation = CausalEstimation(self)
        self.auditor = CausalAuditor(self)
        self._dag = None

    def set_dag(self, edges):
        """
        Manually set the DAG structure as a list of directed edges.
        Example: [('A', 'B'), ('B', 'C')]
        """
        self._dag = edges
        self.discovery.set_dag(edges)

    def get_dag(self, **kwargs):
        """
        Get the current DAG structure.
        """
        if self._dag is None:
            # Fallback to discovering DAG if not set manually
            self._dag = self.discovery.discover(**kwargs)
        return self._dag


    def estimate_ate(self):
        """
        Estimate the Average Treatment Effect (ATE).
        """
        return self.estimation.estimate_ate()

    def estimate_cate(self, new_data=None):
        """
        Estimate the Conditional Average Treatment Effect (CATE) for a population.
        """
        return self.estimation.estimate_cate(new_data)

    def audit_consistency(self, patient, interventions, threshold_conflict=0.01):
        """
        Audit the consistency of the predictive, SCM, and DML layers.
        """
        return self.auditor.audit(patient, interventions, threshold_conflict=threshold_conflict)
