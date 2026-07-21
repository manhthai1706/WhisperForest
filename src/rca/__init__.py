from .engine.scm import SCMEngine
from .engine.mixture_scm import MixtureOfSCMEngine
from .attribution.attributor import Attributor

class RCABranch:
    def __init__(self, parent):
        self.parent = parent
        self.engine = SCMEngine(self)
        self.mixture_engine = MixtureOfSCMEngine(self)
        self.attributor = Attributor(self)

    def fit_scm(self):
        """
        Fit structural equations for all nodes in the DAG.
        """
        self.engine.fit()

    def fit_mixture_scm(self, n_clusters=3, n_iterations=5):
        """
        Fit structural equations for multiple latent causal mechanisms.
        """
        self.mixture_engine.fit(n_clusters=n_clusters, n_iterations=n_iterations)

    def analyze_anomaly(self, anomaly_data, baseline_data, causal_graph=None, method="graph", k_neighbors=10):
        """
        Root Cause Analysis theo ca — độc lập SCM khi dùng method='graph'.

        method='graph'  : World DAG + lệch cohort (mặc định, không cần fit_scm).
        method='intervention' / 'structural' / 'noise' : cần fit_scm() trước.
        """
        if causal_graph is None:
            causal_graph = self.parent.causal.get_dag()

        return self.attributor.attribute(
            anomaly_data, baseline_data, causal_graph, method=method, k_neighbors=k_neighbors
        )



    def plot_attribution(self, attribution_report, save_path=None):
        """
        Plots the attribution report as a beautiful horizontal bar chart representing root causes.
        """
        return self.attributor.plot_attribution(attribution_report, save_path)

