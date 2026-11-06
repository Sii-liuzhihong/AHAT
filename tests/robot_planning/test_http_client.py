"""Real local HTTP roundtrips, including retry, authentication, and redaction."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import unittest

from robot_planning.clients.llm import OpenAICompatibleClient
from robot_planning.repair import fix_cot


class HTTPClientTests(unittest.TestCase):
    def test_repair_and_retry_over_http(self):
        requests = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append((self.path, self.headers.get("Authorization"), body))
                self.send_response(429 if len(requests) == 1 else 200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"choices": [{"message": {"content": "<fixed_cot>fixed</fixed_cot>"}}]}).encode())

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            client = OpenAICompatibleClient(base_url=f"http://127.0.0.1:{server.server_port}/v1",
                                            api_key="synthetic-secret", model="local", max_retries=2)
            result = fix_cot("task", "scene", "old", "bad", client=client)
            self.assertEqual(result["fix_cot"], "fixed")
            self.assertEqual(len(requests), 2)
            path, authorization, body = requests[-1]
            self.assertEqual(path, "/v1/chat/completions")
            self.assertEqual(authorization, "Bearer synthetic-secret")
            self.assertEqual(body["model"], "local")
            self.assertIn("**original_cot**: old", body["messages"][1]["content"])
        finally:
            server.shutdown()
            server.server_close()
            worker.join()

    def test_terminal_error_does_not_echo_server_secrets(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(401)
                self.end_headers()
                self.wfile.write(b"synthetic-secret")

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            response = OpenAICompatibleClient(base_url=f"http://127.0.0.1:{server.server_port}",
                                              model="local", max_retries=3).call([])
            self.assertFalse(response.success)
            self.assertEqual(response.error, "Chat endpoint returned HTTP 401")
            self.assertEqual(response.retry_count, 0)
            self.assertNotIn("synthetic-secret", repr(response))
        finally:
            server.shutdown()
            server.server_close()
            worker.join()


if __name__ == "__main__":
    unittest.main()
