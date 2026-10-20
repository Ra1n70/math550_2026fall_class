"""Watch a learned MinAtar policy in a local browser, with live playback controls."""

from __future__ import annotations

import argparse
import errno
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import urlsplit
import webbrowser

from math550.topics.reinforcement.playback import MinAtarPolicyPlayer
from math550.topics.reinforcement.data import minatar_action_ids
from math550.topics.reinforcement.visualization import MINATAR_CHANNEL_NAMES

from .play_policy import _default_checkpoint, _load_network


UI_FILE = Path(__file__).with_name("ui") / "index.html"


def make_server(
    checkpoints: dict[str, Path],
    algorithm: str = "dqn",
    seed: int = 10_550,
    game: str = "breakout",
    device: str = "auto",
    max_episode_steps: int = 1_000,
    port: int = 0,
) -> ThreadingHTTPServer:
    """Serve one shared episode on loopback, serializing inference and resets.

    Checkpoints are selected at launch, never accepted as paths over HTTP.
    Browser controls advance the real MinAtar environment through the same
    greedy decision function as the existing GIF exporter.
    If the preferred port is occupied, the OS chooses a free loopback port.
    """

    if type(port) is not int or not 0 <= port <= 65_535:
        raise ValueError("port must be an integer between 0 and 65535")
    if game not in MINATAR_CHANNEL_NAMES:
        raise ValueError(f"unsupported MinAtar game: {game}")
    paths = {key: Path(path) for key, path in checkpoints.items() if key in ("dqn", "ppo")}
    if algorithm not in paths:
        raise ValueError(f"no {algorithm} checkpoint configured")
    channels = len(MINATAR_CHANNEL_NAMES[game])
    networks = {}

    def load_player(selected: str, selected_seed: int) -> MinAtarPolicyPlayer:
        if selected not in paths:
            raise ValueError(f"no {selected} checkpoint configured")
        if selected not in networks:
            networks[selected] = _load_network(
                selected, paths[selected], channels, len(minatar_action_ids(game)), game=game
            )
        return MinAtarPolicyPlayer(
            networks[selected], selected, game, selected_seed, max_episode_steps, device
        )

    player = load_player(algorithm, seed)
    lock = threading.Lock()

    def snapshot() -> dict[str, object]:
        return {
            **player.snapshot(),
            "available_algorithms": list(paths),
            "checkpoint": paths[player.algorithm].name,
        }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            # Per-frame requests should not flood the classroom terminal.
            pass

        def send_content(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, status: int, payload: dict[str, object]) -> None:
            self.send_content(
                status,
                json.dumps(payload, allow_nan=False).encode("utf-8"),
                "application/json; charset=utf-8",
            )

        def is_local_request(self) -> bool:
            allowed_hosts = {
                f"127.0.0.1:{self.server.server_port}",
                f"localhost:{self.server.server_port}",
            }
            if self.headers.get("Host") not in allowed_hosts:
                self.send_json(403, {"error": "Use the printed localhost URL."})
                return False
            origin = self.headers.get("Origin")
            if origin and origin not in {f"http://{host}" for host in allowed_hosts}:
                self.send_json(403, {"error": "Requests must come from the local player."})
                return False
            return True

        def do_GET(self) -> None:
            if not self.is_local_request():
                return
            path = urlsplit(self.path).path
            if path in ("/", "/index.html"):
                self.send_content(200, UI_FILE.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/state":
                with lock:
                    self.send_json(200, snapshot())
            elif path == "/favicon.ico":
                self.send_content(204, b"", "image/x-icon")
            else:
                self.send_json(404, {"error": "Unknown player endpoint."})

        def do_POST(self) -> None:
            nonlocal player
            if not self.is_local_request():
                return
            path = urlsplit(self.path).path
            if path not in ("/api/step", "/api/reset"):
                self.send_json(404, {"error": "Unknown player endpoint."})
                return
            try:
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("Use application/json for player controls.")
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 4_096:
                    raise ValueError("The control request must contain a small JSON object.")
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise ValueError("The control request must be a JSON object.")
                with lock:
                    if path == "/api/step":
                        player.step()
                    else:
                        selected = payload.get("algorithm", player.algorithm)
                        if not isinstance(selected, str) or selected not in paths:
                            raise ValueError("Choose an available policy: " + ", ".join(paths))
                        selected_seed = payload.get("seed", player.seed)
                        if type(selected_seed) is not int or not 0 <= selected_seed <= 2**32 - 1:
                            raise ValueError("seed must be an integer between 0 and 4294967295")
                        # Only replace the current episode once loading/reset succeeds.
                        player = load_player(selected, selected_seed)
                    result = snapshot()
                self.send_json(200, result)
            except (ValueError, UnicodeError, TypeError) as error:
                self.send_json(400, {"error": str(error)})
            except (OSError, RuntimeError) as error:
                self.send_json(500, {"error": f"Unable to run this checkpoint: {error}"})

    try:
        return ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError as error:
        if error.errno != errno.EADDRINUSE or port == 0:
            raise
        # Let the OS reserve a free port atomically. Do not kill the existing
        # player, probe consecutive ports, or reload the model to retry.
        return ThreadingHTTPServer(("127.0.0.1", 0), Handler)


def serve_policy_ui(
    algorithm: str = "dqn",
    checkpoint: Path | None = None,
    game: str = "breakout",
    seed: int = 10_550,
    device: str = "auto",
    max_episode_steps: int = 1_000,
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    """Launch the local player and keep it running until Ctrl+C."""

    if checkpoint is None and game != "breakout":
        raise ValueError("Supply --checkpoint for a policy trained on this game.")
    checkpoints = (
        {algorithm: checkpoint}
        if checkpoint is not None
        else {name: _default_checkpoint(name) for name in ("dqn", "ppo")
              if _default_checkpoint(name).is_file()}
    )
    if algorithm not in checkpoints:
        raise FileNotFoundError(
            f"No learned {algorithm} checkpoint found. Run `make quick-reinforcement` "
            "or pass --checkpoint with your trained weights."
        )
    server = make_server(
        checkpoints, algorithm, seed, game, device, max_episode_steps, port
    )
    url = f"http://127.0.0.1:{server.server_port}/"
    if port and server.server_port != port:
        print(
            f"Port {port} is already in use; using available port {server.server_port}.",
            flush=True,
        )
    print(f"MinAtar policy player: {url}", flush=True)
    print("Press Play in the browser. Stop the server with Ctrl+C.", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nPlayer stopped.", flush=True)
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("dqn", "ppo"), default="dqn")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--game", choices=tuple(MINATAR_CHANNEL_NAMES), default="breakout")
    parser.add_argument("--seed", type=int, default=10_550)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--max-episode-steps", type=int, default=1_000)
    parser.add_argument(
        "--port", type=int, default=8765,
        help="Preferred local port (default: 8765); use a free port if busy. 0 selects any free port.",
    )
    parser.add_argument("--no-browser", action="store_true", help="Print the URL without opening a browser")
    args = parser.parse_args()
    try:
        serve_policy_ui(
            args.algorithm, args.checkpoint, args.game, args.seed, args.device,
            args.max_episode_steps, args.port, not args.no_browser,
        )
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Player could not start: {error}\n")


if __name__ == "__main__":
    main()
