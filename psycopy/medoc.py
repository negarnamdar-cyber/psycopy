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

    Manages per-trial connection lifecycle:
    - Connect → TRIGGER → wait → GET_STATUS → disconnect
    """

    PROGRAM_CODES: dict[str, int] = {
        "unified": 192,
        "xlow": 121,
        "low": 83,
        "medium": 82,
        "high": 87,
    }
    GET_STATUS = 0       # temperature / device-state poll
    SELECT_TP = 1
    START = 2
    INTER_CMD_DELAY_SEC = 0.5

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
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.settimeout(self._config.medoc_timeout)
            self._sock.connect((self._config.medoc_ip, self._config.medoc_port))
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
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass  # Ignore errors on close
            finally:
                self._sock = None
        self._state = ConnectionState.DISCONNECTED

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

    def _recv_exact(self, num_bytes: int) -> bytes:
        if self._sock is None:
            raise RuntimeError("Socket not connected - call connect() first")

        chunks: list[bytes] = []
        remaining = num_bytes
        while remaining > 0:
            chunk = self._sock.recv(remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _send_framed_command(self, cmd_id: int, param_bytes: bytes | None = None, tag: str = "") -> bytes:
        if self._sock is None:
            raise RuntimeError("Socket not connected - call connect() first")

        request = self._build_frame(cmd_id, param_bytes)
        logger.debug("Sending Medoc frame %s: %s", tag or cmd_id, request.hex())
        self._sock.sendall(request)

        header = self._recv_exact(4)
        if len(header) != 4:
            raise MedocResponseError(
                response_code=-1,
                raw_bytes=header,
                message=f"Incomplete response header for {tag or cmd_id}",
            )

        body_length = int.from_bytes(header, "big")
        body = self._recv_exact(body_length)
        raw_response = header + body
        logger.debug("Received Medoc frame %s: %s", tag or cmd_id, raw_response.hex())
        if len(body) != body_length:
            raise MedocResponseError(
                response_code=-1,
                raw_bytes=raw_response,
                message=f"Incomplete response body for {tag or cmd_id}",
            )
        return raw_response

    def _send_framed_command_once(
        self, cmd_id: int, param_bytes: bytes | None = None, tag: str = ""
    ) -> bytes:
        self.connect()
        try:
            return self._send_framed_command(cmd_id, param_bytes=param_bytes, tag=tag)
        finally:
            self.disconnect()

    def _parse_response_code(self, raw_response: bytes) -> int:
        if len(raw_response) < 13:
            raise MedocResponseError(
                response_code=-1,
                raw_bytes=raw_response,
                message="Response too short to parse Medoc response code",
            )
        body = raw_response[4:]
        return int.from_bytes(body[7:9], "big")

    def _is_incomplete_response_error(self, exc: MedocResponseError) -> bool:
        return exc.response_code == -1

    def _is_ignorable_socket_error(self, exc: OSError) -> bool:
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

    def send_program(self, pain_condition: str) -> None:
        """Send the configured Medoc program command for a pain condition.

        Uses the same MMS framed protocol as `finilized.py`:
        1. SELECT_TP with the program code
        2. START the selected program

        Raises:
            ValueError: If pain_condition is unknown
            MedocTimeoutError: If response times out
            MedocResponseError: If response indicates failure
            RuntimeError: If socket is not connected
        """
        try:
            program_code = self.PROGRAM_CODES[pain_condition]
        except KeyError as exc:
            valid = ", ".join(sorted(self.PROGRAM_CODES))
            raise ValueError(f"Unknown pain condition {pain_condition!r}; expected one of: {valid}") from exc

        try:
            try:
                select_response = self._send_framed_command_once(
                    self.SELECT_TP,
                    param_bytes=self._u32be(socket.htonl(program_code)),
                    tag=f"SELECT_TP {pain_condition} (A)",
                )
                select_rc = self._parse_response_code(select_response)
                if select_rc != MedocResponseCode.OK:
                    time.sleep(self.INTER_CMD_DELAY_SEC)
                    select_response = self._send_framed_command_once(
                        self.SELECT_TP,
                        param_bytes=self._u32be(program_code),
                        tag=f"SELECT_TP {pain_condition} (B)",
                    )
                    select_rc = self._parse_response_code(select_response)
                if select_rc != MedocResponseCode.OK:
                    raise MedocResponseError(
                        response_code=select_rc,
                        raw_bytes=select_response,
                        message=f"SELECT_TP failed for {pain_condition} with code {select_rc}",
                    )
            except MedocResponseError as exc:
                if self._is_incomplete_response_error(exc):
                    logger.warning(
                        "Ignoring incomplete SELECT_TP response for %s and continuing: %s",
                        pain_condition,
                        exc,
                    )
                else:
                    raise

            time.sleep(self.INTER_CMD_DELAY_SEC)

            try:
                start_response = self._send_framed_command_once(
                    self.START,
                    tag=f"START {pain_condition}",
                )
                start_rc = self._parse_response_code(start_response)
                if start_rc != MedocResponseCode.OK:
                    raise MedocResponseError(
                        response_code=start_rc,
                        raw_bytes=start_response,
                        message=f"START failed for {pain_condition} with code {start_rc}",
                    )
            except MedocResponseError as exc:
                if self._is_incomplete_response_error(exc):
                    logger.warning(
                        "Ignoring incomplete START response for %s and continuing: %s",
                        pain_condition,
                        exc,
                    )
                else:
                    raise

        except socket.timeout:
            raise MedocTimeoutError(
                self._config.medoc_timeout,
                f"Timeout waiting for Medoc framed response from {self._config.medoc_ip}:{self._config.medoc_port}",
            ) from None
        except OSError as exc:
            if self._is_ignorable_socket_error(exc):
                logger.warning(
                    "Ignoring Medoc socket abort for %s and continuing: %s",
                    pain_condition,
                    exc,
                )
                return
            raise

    def send_unified_program(self) -> None:
        """Send the unified program via SELECT_TP + START.

        Keeps the socket open so the caller can continue polling status.
        Must be called while the socket is already connected.

        Raises:
            MedocResponseError: If SELECT_TP or START fails.
            RuntimeError: If socket is not connected.
        """
        program_code = self.PROGRAM_CODES["unified"]
        try:
            select_response = self._send_framed_command(
                self.SELECT_TP,
                param_bytes=self._u32be(socket.htonl(program_code)),
                tag=f"SELECT_TP unified (A)",
            )
            select_rc = self._parse_response_code(select_response)
            if select_rc != MedocResponseCode.OK:
                time.sleep(self.INTER_CMD_DELAY_SEC)
                select_response = self._send_framed_command(
                    self.SELECT_TP,
                    param_bytes=self._u32be(program_code),
                    tag=f"SELECT_TP unified (B)",
                )
                select_rc = self._parse_response_code(select_response)
            if select_rc != MedocResponseCode.OK:
                raise MedocResponseError(
                    response_code=select_rc,
                    raw_bytes=select_response,
                    message=f"SELECT_TP failed for unified with code {select_rc}",
                )

            time.sleep(self.INTER_CMD_DELAY_SEC)

            start_response = self._send_framed_command(
                self.START,
                tag="START unified",
            )
            start_rc = self._parse_response_code(start_response)
            if start_rc != MedocResponseCode.OK:
                raise MedocResponseError(
                    response_code=start_rc,
                    raw_bytes=start_response,
                    message=f"START failed for unified with code {start_rc}",
                )
        except socket.timeout:
            raise MedocTimeoutError(
                self._config.medoc_timeout,
                "Timeout waiting for Medoc framed response during unified program start",
            ) from None

    def poll_status(self) -> dict[str, Any]:
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
            RuntimeError: If socket is not connected.
        """
        raw_response = self._send_framed_command(
            self.GET_STATUS,
            tag="GET_STATUS",
        )
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
