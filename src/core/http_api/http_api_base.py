from enum import Enum
import json

from requests import request


class HttpApiBase:

    def __init__(self, base_url: str):
        self._base_url = base_url

    def _send(self, endpoint: str | Enum, method: str,
              query_params: dict = None, json_data: dict = None,
              rel_path: str = ''):

        # --- build url
        if isinstance(endpoint, Enum):
            endpoint = endpoint.value

        if rel_path and len(rel_path) > 0:
            endpoint = f"{endpoint}/{rel_path}"

        url = f"{self._base_url}/{endpoint}"

        headers = {}

        if json_data:
            json_data = json.dumps(json_data)
            headers['Content-Type'] = 'application/json'

        # send request
        response = request(method, url, headers=headers,
                           params=query_params, data=json_data)

        # check response
        response.raise_for_status()
        return response.json()

    def get(self, endpoint: str | Enum,
            params: dict = None, rel_path: str = '',
            json: dict = {}) -> dict:
        """
        Perform a GET request to the specified endpoint.
        """
        return self._send(endpoint, "GET",
                          query_params=params, json_data=json,
                          rel_path=rel_path)

    def post(self, endpoint: str | Enum,
             query_params: dict = None, rel_path: str = '',
             json: dict = {}) -> dict:
        """
        Perform a POST request to the specified endpoint.
        """
        return self._send(endpoint, "POST", query_params, json, rel_path)
