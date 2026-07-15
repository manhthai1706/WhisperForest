from .engine.scm import SCMEngine
from .attribution.attributor import Attributor

class RCABranch:
    def __init__(self, parent):
        self.parent = parent
        self.engine = SCMEngine(self)
        self.attributor = Attributor(self)

    def fit_scm(self):
        """
        Fit structural equations for all nodes in the DAG.
        """
        self.engine.fit()

    def analyze_anomaly(self, anomaly_data, baseline_data, causal_graph=None, method="intervention", k_neighbors=10):
        """
        Perform Root Cause Analysis on an anomaly dataset compared to baseline data.
        """
        if not self.engine.is_fitted:
            self.fit_scm()
            
        if causal_graph is None:
            causal_graph = self.parent.causal.get_dag()
            
        return self.attributor.attribute(anomaly_data, baseline_data, causal_graph, method=method, k_neighbors=k_neighbors)



    def plot_attribution(self, attribution_report, save_path=None):
        """
        Plots the attribution report as a beautiful horizontal bar chart representing root causes.
        """
        return self.attributor.plot_attribution(attribution_report, save_path)

