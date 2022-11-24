"""URL helpers."""
from urllib.parse import urlparse


def redact_url_password(url: str) -> str:
    """Remove password from the URL.

    Designed to sanitize PSQL connection URLs in logs.

    :param url: URL as a string
    :return: URL where password is replaced with ???
    """
    # https://stackoverflow.com/a/46905953/315168
    parsed = urlparse(url)
    if parsed.port:
        replaced = parsed._replace(netloc="{}:{}@{}:{}".format(parsed.username, "???", parsed.hostname, parsed.port))
        return replaced.geturl() # 'https://user:???@example.com/path?key=value#hash
    else:
        replaced = parsed._replace(netloc="{}:{}@{}".format(parsed.username, "???", parsed.hostname))
        return replaced.geturl()


def get_url_domain(url: str) -> str:
    """Redact URL so that only domain is displayed.

    Some services e.g. infura use path as an API key.
    """
    parsed = urlparse(url)
    return parsed.hostname
