from collections import deque, Counter


class ClusterSmoother:
    def __init__(self, history=7, hold_last_valid=True):
        self.history = int(history)
        self.hold_last_valid = bool(hold_last_valid)
        self.buf = deque(maxlen=self.history)
        self.last_cluster = None

    def update(self, cluster):
        if cluster is None:
            return self.last_cluster, False

        if cluster == -1 and self.hold_last_valid and self.last_cluster is not None:
            cluster = self.last_cluster

        self.buf.append(cluster)
        counts = Counter(self.buf)
        smoothed = counts.most_common(1)[0][0]

        changed = smoothed != self.last_cluster
        self.last_cluster = smoothed
        return smoothed, changed
