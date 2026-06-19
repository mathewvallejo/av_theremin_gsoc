from pythonosc.udp_client import SimpleUDPClient


class AVGestureOSCSender:
    def __init__(self, host="127.0.0.1", port=9000, prefix="/av_gesture"):
        self.client = SimpleUDPClient(host, int(port))
        self.prefix = prefix.rstrip("/")

    def send(self, path, *values):
        self.client.send_message(f"{self.prefix}{path}", list(values) if len(values) > 1 else (values[0] if values else 1))

    def send_hand_meta(self, meta):
        self.send("/hand/num_hands", int(meta.get("num_hands", 0)))
        self.send("/hand/right/present", int(meta.get("right_present", 0)))
        self.send("/hand/left/present", int(meta.get("left_present", 0)))

    def send_selected_landmarks(self, meta):
        index = {
            "wrist": 0,
            "thumb_tip": 4,
            "index_tip": 8,
            "middle_tip": 12,
            "ring_tip": 16,
            "pinky_tip": 20,
        }
        for side in ["right", "left"]:
            vec = meta.get(side)
            if vec is None:
                continue
            pts = vec.reshape(21, 3)
            for name, i in index.items():
                self.send(f"/hand/{side}/{name}", float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2]))

    def send_full_landmarks(self, meta):
        for side in ["right", "left"]:
            vec = meta.get(side)
            if vec is not None:
                self.send(f"/hand/{side}/landmarks", *[float(x) for x in vec.tolist()])

    def send_gesture(self, cluster, confidence, name, changed):
        self.send("/gesture/cluster", int(cluster))
        self.send("/gesture/confidence", float(confidence))
        self.send("/gesture/name", str(name))
        self.send("/gesture/changed", int(bool(changed)))

    def send_latent(self, z):
        self.send("/latent", *[float(x) for x in z.tolist()])

    def send_motion(self, energy, window_ready):
        self.send("/motion/energy", float(energy))
        self.send("/motion/window_ready", int(bool(window_ready)))
