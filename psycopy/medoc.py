"""Medoc communication exceptions and client.

Uses the MMS framed protocol:
  Frame = [4-byte big-endian length] + [body]
  Body  = [4-byte timestamp BE] + [1-byte command_id] + [parameters / response data]

GET_STATUS response body layout (after 4-byte length header):
  bytes 0-3   : timestamp (big-endian)
  byte  4     : command_id
  byte  5     : system_state
  byte  6     : test_state
  bytes 7-8   : response code (big-endian)
  bytes 9-12  : tms (big-endian)
  bytes 13-14 : temperature as signed little-endian 16-bit int / 100
  byte  17    : ttl (if present)
"""

from __future__ import annotations

import logging
import socket
import time
from enum import IntEnum
from typing import Any

from psycopy.config import MedocConfig
from psycopy.models import MedocResponseCode

logger = logging.getLogger(__name__)


class MedocConnectionError(RuntimeError):
    """Raised when connection to Medoc device fails."""

    def __init__(self, ip: str, port: int, message: str = ""):
        self.ip = ip
        self.port = port
        if message:
            super().__init__(message)
        else:
            super().__init__(f"Connection refused to {ip}:{port}")


class MedocTimeoutError(RuntimeError):
    """Raised when Medoc communication times out."""

    def __init__(self, timeout: float, message: str = ""):
        self.timeout = timeout
        if message:
            super().__init__(message)
        else:
            super().__init__(f"Timeout after {timeout}s waiting for response")


class MedocResponseError(RuntimeError):
    """Raised when Medoc returns an invalid response."""

    def __init__(self, response_code: int, raw_bytes: bytes | None = None, message: str = ""):
        self.response_code = response_code
        self.raw_bytes = raw_bytes
        if message:
            super().__init__(message)
        else:
            super().__init__(f"Invalid response: code={response_code}")


class ConnectionState(IntEnum):
    """TCP connection state for MedocClient."""

    DISCONNECTED = 0
    CONNECTED = 1
    ERROR = 2


class MedocClient:
    """TCP socket client for Medoc thermode device communication.

    Manages the unified Medoc thermal program:
    - Reachability check -> STOP/ready -> SELECT_TP unified -> START
    - Each MMS command uses its own short TCP connection, matching Medoc MMS behavior.
    """

    UNIFIED_PROGRAM_CODE = 192  # Binary 11000000
    UNIFIED_PROGRAM_LABEL = "unified"
    GET_STATUS = 0       # temperature / device-state poll
    SELECT_TP = 1
    START = 2
    STOP = 5
    INTER_CMD_DELAY_SEC = 0.5
    COMMAND_RESPONSE_TIMEOUT_SEC = 2.0

    def __init__(self, config: MedocConfig) -> None:
        self._config = config
        self._state = ConnectionState.DISCONNECTED
        self._sock: socket.socket | None = None

    @property
    def state(self) -> ConnectionState:
        return self._state

    def connect(self) -> None:
        """Establish TCP connection to Medoc device.

        Raises:
            MedocConnectionError: If connection refused
            MedocTimeoutError: If connection times out
        """
        try:
            self._close_socket_only()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.settimeout(self._config.medoc_timeout)
            sock.connect((self._config.medoc_ip, self._config.medoc_port))
            sock.close()
            self._state = ConnectionState.CONNECTED

        except socket.timeout:
            self._state = ConnectionState.ERROR
            if self._sock:
                self._sock.close()
                self._sock = None
            raise MedocTimeoutError(
                self._config.medoc_timeout,
                f"Connection timeout after {self._config.medoc_timeout}s to {self._config.medoc_ip}:{self._config.medoc_port}",
            ) from None

        except ConnectionRefusedError:
            self._state = ConnectionState.ERROR
            if self._sock:
                self._sock.close()
                self._sock = None
            raise MedocConnectionError(
                self._config.medoc_ip,
                self._config.medoc_port,
            ) from None

        except OSError as e:
            self._state = ConnectionState.ERROR
            if self._sock:
                self._sock.close()
                self._sock = None
            if "timed out" in str(e).lower() or "timeout" in str(e).lower():
                raise MedocTimeoutError(
                    self._config.medoc_timeout,
                    f"Connection timeout after {self._config.medoc_timeout}s to {self._config.medoc_ip}:{self._config.medoc_port}",
                ) from None
            raise MedocConnectionError(
                self._config.medoc_ip,
                self._config.medoc_port,
                f"Connection failed: {e}",
            ) from None

    def disconnect(self) -> None:
        """Close TCP connection and reset state.

        Safe to call multiple times or when not connected.
        """
        self._close_socket_only()
        self._state = ConnectionState.DISCONNECTED

    def _close_socket_only(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass  # Ignore errors on close
            finally:
                self._sock = None

    def __enter__(self) -> MedocClient:
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()

    def _u32be(self, value: int) -> bytes:
        return (value & 0xFFFFFFFF).to_bytes(4, "big", signed=False)

    def _build_frame(self, cmd_id: int, param_bytes: bytes | None = None) -> bytes:
        timestamp_be = self._u32be(socket.htonl(int(time.time())))
        body = timestamp_be + cmd_id.to_bytes(1, "big")
        if param_bytes is not None:
            body += param_bytes
        return self._u32be(len(body)) + body

    def _recv_exact_from(self, sock: socket.socket, num_bytes: int) -> bytes:
        chunks: list[bytes] = []
        remaining = num_bytes
        while remaining > 0:
            chunk = sock.recv(remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _recv_exact(self, num_bytes: int) -> bytes:
        if self._sock is None:
            raise RuntimeError("Socket not connected - call connect() first")
        return self._recv_exact_from(self._sock, num_bytes)

    def _read_framed_response(
        self,
        sock: socket.socket,
        tag: str,
        allow_incomplete: bool = False,
    ) -> bytes:
        header = self._recv_exact_from(sock, 4)
        if len(header) != 4:
            if allow_incomplete:
                logger.debug("Incomplete Medoc response header for %s: %s", tag, header.hex())
                return header
            raise MedocResponseError(
                response_code=-1,
                raw_bytes=header,
                message=f"Incomplete response header for {tag}",
            )

        body_length = int.from_bytes(header, "big")
        body = self._recv_exact_from(sock, body_length)
        raw_response = header + body
        logger.debug("Received Medoc frame %s: %s", tag, raw_response.hex())
        if len(body) != body_length:
            if allow_incomplete:
                logger.debug(
                    "Incomplete Medoc response body for %s: expected=%d actual=%d",
                    tag,
                    body_length,
                    len(body),
                )
                return raw_response
            raise MedocResponseError(
                response_code=-1,
                raw_bytes=raw_response,
                message=f"Incomplete response body for {tag}",
            )
        return raw_response

    def _send_framed_command(self, cmd_id: int, param_bytes: bytes | None = None, tag: str = "") -> bytes:
        if self._sock is None:
            raise RuntimeError("Socket not connected - call connect() first")

        request = self._build_frame(cmd_id, param_bytes)
        logger.debug("Sending Medoc frame %s: %s", tag or cmd_id, request.hex())
        self._sock.sendall(request)
        return self._read_framed_response(self._sock, tag or str(cmd_id))

    def _send_framed_command_once(
        self,
        cmd_id: int,
        param_bytes: bytes | None = None,
        tag: str = "",
        allow_incomplete: bool = False,
    ) -> bytes:
        request = self._build_frame(cmd_id, param_bytes)
        label = tag or str(cmd_id)
        logger.debug("Sending Medoc frame %s: %s", label, request.hex())

        # The validated MMS script opens a fresh TCP connection per command.
        self._close_socket_only()
        try:
            with socket.create_connection(
                (self._config.medoc_ip, self._config.medoc_port),
                timeout=self._config.medoc_timeout,
            ) as sock:
                sock.settimeout(min(self._config.medoc_timeout, self.COMMAND_RESPONSE_TIMEOUT_SEC))
                sock.sendall(request)
                return self._read_framed_response(sock, label, allow_incomplete=allow_incomplete)
        except socket.timeout:
            if allow_incomplete:
                logger.debug("Timed out waiting for Medoc response to %s", label)
                return b""
            raise MedocTimeoutError(
                self._config.medoc_timeout,
                f"Timeout waiting for Medoc framed response from {self._config.medoc_ip}:{self._config.medoc_port}",
            ) from None
        except ConnectionRefusedError:
            self._state = ConnectionState.ERROR
            raise MedocConnectionError(
                self._config.medoc_ip,
                self._config.medoc_port,
            ) from None
        except OSError as exc:
            if allow_incomplete and self._is_incomplete_socket_error(exc):
                logger.debug("Medoc closed connection while waiting for %s: %s", label, exc)
                return b""
            self._state = ConnectionState.ERROR
            raise MedocConnectionError(
                self._config.medoc_ip,
                self._config.medoc_port,
                f"Medoc command {label} failed: {exc}",
            ) from None

    def _parse_response_code(self, raw_response: bytes) -> int:
        if len(raw_response) < 13:
            raise MedocResponseError(
                response_code=-1,
                raw_bytes=raw_response,
                message="Response too short to parse Medoc response code",
            )
        body = raw_response[4:]
        return int.from_bytes(body[7:9], "big")

    def _parse_response_code_or_none(self, raw_response: bytes) -> int | None:
        if len(raw_response) < 13:
            return None
        body = raw_response[4:]
        return int.from_bytes(body[7:9], "big")

    def _is_incomplete_socket_error(self, exc: OSError) -> bool:
        message = str(exc).lower()
        return any(
            pattern in message
            for pattern in (
                "established connection was aborted",
                "forcibly closed by the remote host",
                "connection reset by peer",
                "broken pipe",
            )
        )

    def _select_unified_program(self) -> bool:
        param_a = self._u32be(socket.htonl(self.UNIFIED_PROGRAM_CODE))
        raw = self._send_framed_command_once(
            self.SELECT_TP,
            param_bytes=param_a,
            tag=f"SELECT_TP {self.UNIFIED_PROGRAM_LABEL} (A)",
            allow_incomplete=True,
        )
        rc = self._parse_response_code_or_none(raw)
        if rc == MedocResponseCode.OK:
            logger.info("SELECT_TP %s OK (A)", self.UNIFIED_PROGRAM_LABEL)
            return True

        time.sleep(self.INTER_CMD_DELAY_SEC)

        param_b = self._u32be(self.UNIFIED_PROGRAM_CODE)
        raw = self._send_framed_command_once(
            self.SELECT_TP,
            param_bytes=param_b,
            tag=f"SELECT_TP {self.UNIFIED_PROGRAM_LABEL} (B)",
            allow_incomplete=True,
        )
        rc = self._parse_response_code_or_none(raw)
        if rc == MedocResponseCode.OK:
            logger.info("SELECT_TP %s OK (B)", self.UNIFIED_PROGRAM_LABEL)
            return True

        logger.warning("SELECT_TP %s failed; last response code=%s", self.UNIFIED_PROGRAM_LABEL, rc)
        return False

    def _start_selected_program(self) -> bool:
        raw = self._send_framed_command_once(
            self.START,
            tag=f"START {self.UNIFIED_PROGRAM_LABEL}",
            allow_incomplete=True,
        )
        rc = self._parse_response_code_or_none(raw)
        if rc not in (None, MedocResponseCode.OK):
            logger.warning("START %s response code=%s", self.UNIFIED_PROGRAM_LABEL, rc)
            return False

        time.sleep(0.2)
        try:
            status = self.poll_status(tag=f"VERIFY_START({self.UNIFIED_PROGRAM_LABEL})")
        except Exception as exc:
            logger.warning(
                "Could not verify unified program status after START; continuing anyway: %s",
                exc,
            )
            return True

        if status.get("test_state") == 1:
            logger.info("Unified program verified running (test_state=1)")
            return True

        logger.warning(
            "Unified program test_state=%s after START",
            status.get("test_state"),
        )
        return False

    def _stop_to_ready(self) -> None:
        try:
            self._send_framed_command_once(
                self.STOP,
                tag=f"STOP {self.UNIFIED_PROGRAM_LABEL}",
                allow_incomplete=True,
            )
        except Exception as exc:
            logger.debug("STOP before unified start did not complete: %s", exc)

        deadline = time.time() + 10.0
        while time.time() < deadline:
            time.sleep(0.2)
            try:
                status = self.poll_status(tag=f"WAIT_READY({self.UNIFIED_PROGRAM_LABEL})")
            except Exception as exc:
                logger.debug("WAIT_READY poll failed: %s", exc)
                continue

            if status.get("test_state") in (0, 3) and status.get("device_state") in (0, 1):
                return

        logger.warning("Medoc still not READY after STOP; proceeding anyway")

    def send_unified_program(self) -> None:
        """Send the unified program via SELECT_TP + START.

        Uses the same MMS command flow as the validated standalone script,
        with the current experiment's unified program code (192 / 11000000).

        Raises:
            MedocResponseError: If SELECT_TP or START fails.
        """
        self._stop_to_ready()

        if not self._select_unified_program():
            try:
                self.poll_status(tag="GET_STATUS(after-select-fail)")
            except Exception as exc:
                logger.warning("Could not poll status after SELECT_TP failure: %s", exc)
            raise MedocResponseError(
                response_code=-1,
                raw_bytes=None,
                message=f"SELECT_TP failed for {self.UNIFIED_PROGRAM_LABEL}",
            )

        time.sleep(self.INTER_CMD_DELAY_SEC)

        if not self._start_selected_program():
            self._stop_to_ready()
            if not self._start_selected_program():
                raise MedocResponseError(
                    response_code=-1,
                    raw_bytes=None,
                    message=f"START failed for {self.UNIFIED_PROGRAM_LABEL}",
                )

    def stop_unified_program(self) -> None:
        """Stop the current Medoc program and wait briefly for READY/IDLE."""
        self._stop_to_ready()

    def poll_status(self, tag: str = "GET_STATUS") -> dict[str, Any]:
        """Poll Medoc device for current temperature and state.

        Sends command 0 (GET_STATUS) using the framed binary protocol and
        parses the response per the MMS specification.

        Returns:
            Dict with keys:
                - response_code (int)
                - temperature_celsius (float)
                - device_state (int)
                - test_state (int)
                - raw_bytes (bytes)

        Raises:
            MedocResponseError: If the response is too short.
        """
        raw_response = self._send_framed_command_once(
            self.GET_STATUS,
            tag=tag,
            allow_incomplete=True,
        )
        if len(raw_response) < 16:
            logger.debug(
                "poll_status incomplete response (%d bytes), returning partial",
                len(raw_response),
            )
            return {
                "timestamp": 0,
                "command_id": self.GET_STATUS,
                "response_code": -1,
                "temperature_celsius": None,
                "device_state": None,
                "test_state": None,
                "tms": 0,
                "ttl": 0,
                "raw_bytes": raw_response,
            }
        return self._parse_status(raw_response)

    def _parse_status(self, raw_response: bytes) -> dict[str, Any]:
        """Parse a GET_STATUS response into a dictionary.

        Frame layout (after the 4-byte big-endian length header):
          bytes 0-3   : timestamp
          byte  4     : command_id
          byte  5     : system_state
          byte  6     : test_state
          bytes 7-8   : response code (big-endian)
          bytes 9-12  : tms (big-endian)
          bytes 13-14 : temperature as signed little-endian 16-bit int / 100
          byte  17    : ttl (if present)
        """
        if len(raw_response) < 16:  # 4 header + 12 body minimum
            raise MedocResponseError(
                response_code=-1,
                raw_bytes=raw_response,
                message="Response too short to parse Medoc status",
            )

        body = raw_response[4:]
        off = 0

        timestamp = int.from_bytes(body[off : off + 4], "big")
        off += 4

        command_id = body[off]
        off += 1

        system_state = body[off]
        off += 1

        test_state = body[off]
        off += 1

        response_code = int.from_bytes(body[off : off + 2], "big")
        off += 2

        tms = int.from_bytes(body[off : off + 4], "big")
        off += 4

        temperature = int.from_bytes(body[off : off + 2], "little", signed=True) / 100.0
        off += 2

        ttl = body[off + 3] if off + 3 < len(body) else 0

        return {
            "timestamp": timestamp,
            "command_id": command_id,
            "response_code": response_code,
            "temperature_celsius": temperature,
            "device_state": system_state,
            "test_state": test_state,
            "tms": tms,
            "ttl": ttl,
            "raw_bytes": raw_response,
        }
