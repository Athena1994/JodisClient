from pathlib import Path
from typing import Callable
import aiohttp
import requests


class WebRessource:

    class ContentType:
        STREAM = "application/octet-stream"

    def __init__(self, url: str, chunk_size: int = 1024):
        self._content = None

        self._total_size = -1
        self._downloaded = 0

        self._chunk_size = chunk_size

        self._url = url
        self._callback = None

    @property
    def finished(self) -> bool:
        """
        Check if the download is finished.
        :return: True if the download is finished, False otherwise.
        """
        return self._downloaded == self._total_size

    @property
    def callback(self) -> Callable[[int], None]:
        """
        Get the callback function.
        :return: The callback function.
        """
        return self._callback

    @callback.setter
    def callback(self, callback: Callable[[int], None]):
        """
        Register a callback function to be called when the download is complete.
        :param callback: The callback function.
        """
        self._callback = callback

    @property
    def percent(self) -> int:
        """
        Get the download progress percentage.
        :return: The download progress percentage.
        """
        if self._total_size == 0:
            return 0
        return int((self._downloaded / self._total_size) * 100)

    @property
    def content(self) -> bytes:
        """
        Get the downloaded content.
        :return: The downloaded content.
        """
        if not self.finished:
            raise ValueError("Download not finished")
        return self._content

    def _prepare(self):
        response = requests.head(self._url)

        if response.status_code != 200:
            raise ValueError(f"Failed to get headers from {self._url}")

        if 'content-length' not in response.headers:
            raise ValueError(f"Content length not found in headers for"
                             f"{self._url}")
        self._total_size = int(response.headers['content-length'])
        if self._total_size == 0:
            raise ValueError(f"Content length is 0 for {self._url}")

        self._content = b''

        if 'content-type' not in response.headers:
            raise ValueError(f"Content type not found in headers for"
                             f"{self._url}")

        return self._prase_content_type(response.headers['content-type'])

    def _prase_content_type(self, content_type: str):
        if 'application/octet-stream' == WebRessource.ContentType.STREAM:
            return WebRessource.ContentType.STREAM
        raise ValueError(f"Unsupported content type: {content_type}")

    async def _download_chunk(self, session: aiohttp.ClientSession,
                              start: int, end: int):
        """
        Download a chunk of the file.
        :param session: The aiohttp session.
        :param start: The start byte of the chunk.
        :param end: The end byte of the chunk.
        :return: The downloaded chunk.
        """
        headers = {'Range': f'bytes={start}-{end}'}
        async with session.get(self._url, headers=headers) as response:
            if response.status != 206:
                raise ValueError(f"Failed to download chunk {start}-{end} "
                                 f"from {self._url}\n{response}",)
            return await response.read()

    async def download(self):
        """
        Download the content from the URL.
        :return: The downloaded content.
        """

        if self._content is not None:
            raise ValueError("Download may only be called once")

        self._prepare()

        async with aiohttp.ClientSession() as session:
            async with session.get(self._url) as response:
                self._content = await response.read()
                self._downloaded = len(self._content)
                response.raise_for_status()

            # while not self.finished:
            #     chunk_size = min(self._total_size - self._downloaded,
            #                      self._chunk_size)
            #     start = self._downloaded
            #     end = start + chunk_size - 1

            #     chunk = await self._download_chunk(session, start, end)
            #     self._content += chunk
            #     self._downloaded += len(chunk)

            #     if self._callback:
            #         self._callback(self.percent)

    def write_to_file(self, path: Path):
        """
        Write the downloaded content to a file.
        :param path: The path to the file.
        """
        if not self.finished:
            raise ValueError("Download not finished")

        with open(path, 'wb') as f:
            f.write(self._content)
