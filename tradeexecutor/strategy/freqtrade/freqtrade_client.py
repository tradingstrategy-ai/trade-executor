"""REST API client for Freqtrade instances."""

import logging

import requests

logger = logging.getLogger(__name__)


class FreqtradeClient:
    def __init__(
        self,
        api_url: str,
        api_username: str,
        api_password: str,
        timeout: float = 5.0,
    ):
        self.api_url = api_url.rstrip("/")
        self.api_username = api_username
        self.api_password = api_password
        self.timeout = timeout
        self._token = None
        self.session = requests.Session()

    def _get_jwt_token(self) -> str:
        if self._token:
            return self._token

        url = f"{self.api_url}/api/v1/login"
        payload = {
            "username": self.api_username,
            "password": self.api_password,
        }

        
        response = self.session.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        self._token = data.get("access_token")

        if not self._token:
            raise ValueError("No access token in Freqtrade login response")

        return self._token

    def _make_request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> dict:
        url = f"{self.api_url}{endpoint}"
        kwargs.setdefault("timeout", self.timeout)

        # Ensure we have a valid token
        token = self._get_jwt_token()

        # Add JWT authorization header
        if "headers" not in kwargs:
            kwargs["headers"] = {}
        kwargs["headers"]["Authorization"] = f"Bearer {token}"

        response = self.session.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()

    def get_balance(self) -> dict:
        """Query /api/v1/balance endpoint.

        Returns:
            Dict with keys: total, free, used (amounts in reserve currency)

        Raises:
            requests.RequestException: If API call fails
        """
        return self._make_request("GET", "/api/v1/balance")

    def get_status(self) -> dict:
        """Query /api/v1/status for bot and trade status.

        Returns:
            Dict with current bot status and open trades

        Raises:
            requests.RequestException: If API call fails
        """
        return self._make_request("GET", "/api/v1/status")

    def get_performance(self) -> dict:
        """Query /api/v1/performance for trade performance.

        Returns:
            Dict with per-pair trade performance metrics

        Raises:
            requests.RequestException: If API call fails
        """
        return self._make_request("GET", "/api/v1/performance")

