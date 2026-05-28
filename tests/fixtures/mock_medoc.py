"""Mock Medoc TCP socket server for integration testing.

Provides a threaded socketserver that simulates the Medoc thermode device.
Responds to configured Medoc program command bytes.

Usage:
    with MockMedocServer(port=5000) as server:
        client = MedocClient(MedocConfig(medoc_ip="localhost", medoc_port=5000))
        client.connect()
        client.send_unified_program()  # Selects and starts the unified program
        client.disconnect()
"""

from __future__ import annotations

import logging
import socket
import threading
from typing import Callable

logger = logging.getLogger(__name__)


class MockMedocHandler:
    """Handler for Medoc protocol commands.

    Responds to:
    - Program command bytes: Returns OK (0x00)

    Attributes:
        response_delay: Optional delay before responding (for timeout testing)
    """

    # Command codes
    CMD_BASELINE = 0x79
    CMD_LOW = 0x53
    CMD_MEDIUM = 0x52
    CMD_HIGH = 0x57
    VALID_COMMANDS = {CMD_BASELINE, CMD_LOW, CMD_MEDIUM, CMD_HIGH}

    # Response codes
    RESP_OK = 0x00
    RESP_ILLEGAL_PARAMETER = 0x01
    RESP_ILLEGAL_STATE = 0x02
    RESP_NOT_PROPER_STATE = 0x03

    def __init__(
        self,
        response_delay: float | None = None,
        trigger_response: int | None = None,
    ):
        """Initialize mock handler.

        Args:
            response_delay: Delay in seconds before responding (for timeout tests)
            trigger_response: Override trigger response code (default: OK)
        """
        self.response_delay = response_delay
        self.trigger_response = trigger_response if trigger_response is not None else self.RESP_OK

        self._commands_received: list[int] = []
        self._is_running = False

    def handle_command(self, command_byte: int) -> bytes:
        """Handle a single command byte and return response.

        Args:
            command_byte: Command received from client

        Returns:
            Response bytes to send back to client
        """
        self._commands_received.append(command_byte)

        if command_byte in self.VALID_COMMANDS:
            return self._handle_trigger()
        else:
            # Unknown command - return error
            return bytes([self.RESP_ILLEGAL_PARAMETER])

    def _handle_trigger(self) -> bytes:
        """Handle TRIGGER command (0x04).

        Returns:
            Single byte response code
        """
        return bytes([self.trigger_response])

    @property
    def commands_received(self) -> list[int]:
        """List of command bytes received."""
        return self._commands_received.copy()


class MockMedocServer:
    """Threaded TCP socket server that simulates Medoc thermode device.

    Uses socketserver.ThreadingTCPServer for handling concurrent connections.
    Provides start/stop lifecycle methods and thread-safe startup detection.

    Example:
        >>> server = MockMedocServer(port=5000)
        >>> server.start()
        >>> # ... run tests connecting to localhost:5000 ...
        >>> server.stop()

    Attributes:
        port: Port to listen on
        host: Host to bind to (default: localhost)
        handler: MockMedocHandler instance for generating responses
        garbage_mode: If True, sends garbage bytes instead of proper responses
    """

    def __init__(
        self,
        port: int = 5000,
        host: str = "localhost",
        handler: MockMedocHandler | None = None,
        garbage_mode: bool = False,
    ):
        """Initialize mock server.

        Args:
            port: Port to listen on
            host: Host to bind to
            handler: Custom handler instance (default: creates new MockMedocHandler())
            garbage_mode: If True, sends random garbage bytes for response
        """
        self.port = port
        self.host = host
        self.handler = handler or MockMedocHandler()
        self.garbage_mode = garbage_mode

        self._server_socket: socket.socket | None = None
        self._server_thread: threading.Thread | None = None
        self._is_running = False
        self._lock = threading.Lock()
        self._ready_event = threading.Event()

    def start(self, timeout: float = 5.0) -> None:
        """Start the mock server in a background thread.

        Args:
            timeout: Maximum time to wait for server to be ready

        Raises:
            RuntimeError: If server fails to start within timeout
        """
        with self._lock:
            if self._is_running:
                return

            self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._server_socket.bind((self.host, self.port))
            self._server_socket.listen(5)

            self._is_running = True
            self._ready_event.clear()

            self._server_thread = threading.Thread(target=self._run_server, daemon=True)
            self._server_thread.start()

        # Wait for server to be ready
        if not self._ready_event.wait(timeout):
            self.stop()
            raise RuntimeError(f"Mock Medoc server failed to start within {timeout}s")

        logger.debug("Mock Medoc server started on %s:%d", self.host, self.port)

    def _run_server(self) -> None:
        """Server loop running in background thread."""
        self._ready_event.set()
        logger.debug("Server thread started, listening for connections")

        while self._is_running:
            try:
                if self._server_socket is None:
                    break
                self._server_socket.settimeout(0.5)
                client_socket, client_addr = self._server_socket.accept()
                logger.debug("Connection from %s:%d", client_addr[0], client_addr[1])
                self._handle_client(client_socket)
            except socket.timeout:
                continue
            except (OSError, AttributeError):
                break

        logger.debug("Server thread exiting")

    def _handle_client(self, client_socket: socket.socket) -> None:
        """Handle a single client connection.

        Reads command byte and sends response.
        Handles garbage_mode for testing invalid responses.

        Args:
            client_socket: Connected client socket
        """
        try:
            client_socket.settimeout(self.handler.response_delay or 5.0)

            while True:
                try:
                    command_byte_bytes = client_socket.recv(1)
                    if not command_byte_bytes:
                        break

                    command_byte = command_byte_bytes[0]
                    logger.debug("Received command byte: 0x%02x", command_byte)

                    if self.handler.response_delay:
                        import time

                        time.sleep(self.handler.response_delay)

                    if self.garbage_mode:
                        # Send garbage bytes
                        response = b"\xff\xfe\xfd\xfc\xfb\xfa\xf9\xf8"
                    else:
                        response = self.handler.handle_command(command_byte)

                    client_socket.sendall(response)
                    logger.debug("Sent response: %r", response)

                except socket.timeout:
                    logger.debug("Client timeout, closing connection")
                    break
                except ConnectionResetError:
                    logger.debug("Connection reset by client")
                    break

        finally:
            client_socket.close()

    def stop(self) -> None:
        """Stop the mock server and clean up resources."""
        with self._lock:
            self._is_running = False
            if self._server_socket:
                try:
                    self._server_socket.close()
                except OSError:
                    pass
                self._server_socket = None

        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=2.0)

        logger.debug("Mock Medoc server stopped")

    def __enter__(self) -> "MockMedocServer":
        """Context manager entry - start server."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - stop server."""
        self.stop()

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._is_running


class SlowMockMedocServer(MockMedocServer):
    """Mock server with configurable delays for timeout testing.

    Specialized server that adds delays before sending responses,
    useful for testing client timeout handling.
    """

    def __init__(
        self,
        port: int = 5000,
        host: str = "localhost",
        response_delay: float = 10.0,
    ):
        """Initialize slow server.

        Args:
            port: Port to listen on
            host: Host to bind to
            response_delay: Delay in seconds before each response
        """
        handler = MockMedocHandler(response_delay=response_delay)
        super().__init__(port=port, host=host, handler=handler)


class GarbageMockMedocServer(MockMedocServer):
    """Mock server that returns garbage bytes for testing invalid response handling.

    Useful for testing MedocResponseError handling.
    """

    def __init__(
        self,
        port: int = 5000,
        host: str = "localhost",
    ):
        """Initialize garbage server.

        Args:
            port: Port to listen on
            host: Host to bind to
        """
        super().__init__(port=port, host=host, garbage_mode=True)


class ErrorMockMedocServer(MockMedocServer):
    """Mock server that returns error codes instead of OK.

    Useful for testing MedocResponseError handling.
    """

    def __init__(
        self,
        port: int = 5000,
        host: str = "localhost",
        trigger_error_code: int = MockMedocHandler.RESP_ILLEGAL_PARAMETER,
        temperature_celsius: float = 100.0,
    ):
        """Initialize error server.

        Args:
            port: Port to listen on
            host: Host to bind to
            trigger_error_code: Error code to return for TRIGGER (default: ILLEGAL_PARAMETER)
            temperature_celsius: Temperature for GET_STATUS response
        """
        handler = MockMedocHandler(
            trigger_response=trigger_error_code,
            temperature_celsius=temperature_celsius,
        )
        super().__init__(port=port, host=host, handler=handler)
