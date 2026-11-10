"""Download a URL safely, for "add this link to my notebook".

    data, final_url, content_type = fetch(url, max_bytes=..., timeout=..., max_redirects=...)

Safety rules (this runs on a server, so a pasted URL must not reach the server's own network):
  - http and https only.
  - The hostname is resolved first, and EVERY address is checked: private (10.x,
    192.168.x, ...), loopback (127.x, ::1), link-local (169.254.x, e.g. cloud
    metadata), multicast, reserved and unspecified addresses are refused.
  - The connection goes to the address that was checked (not a second DNS
    lookup), so a hostname can't switch to an internal address in between
    ("DNS rebinding"). HTTPS still verifies the certificate for the hostname.
  - Redirects are followed by hand, at most max_redirects, each one re-checked.
  - The body is read in pieces and refused as soon as it passes max_bytes.
  - Every socket operation has a timeout, and the whole download has a deadline.

Errors raise FetchError with a message that can be shown to the user.
"""

import http.client
import ipaddress
import socket
import ssl
import time
from urllib.parse import urljoin, urlsplit

USER_AGENT = "rag-app/0.1 (document Q&A)"


class FetchError(Exception):
    pass


def _check_ip(ip_text: str, host: str) -> None:
    ip = ipaddress.ip_address(ip_text)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast
            or ip.is_reserved or ip.is_unspecified):
        raise FetchError(f"refusing to fetch {host}: it points to a private or local address ({ip})")


def resolve_public(url: str) -> tuple[str, int, str]:
    """Validate a URL and return (scheme, port, checked_ip). Raises FetchError."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise FetchError(f"only http:// and https:// links are supported, not {parts.scheme or 'no scheme'}")
    if not parts.hostname:
        raise FetchError("the link has no host name")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(parts.hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        raise FetchError(f"could not find the host {parts.hostname}")
    addresses = sorted({info[4][0] for info in infos})
    for address in addresses:
        _check_ip(address, parts.hostname)
    return parts.scheme, port, addresses[0]


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """Connects to a pre-checked IP while sending the real Host header."""

    def __init__(self, host, ip, **kwargs):
        super().__init__(host, **kwargs)
        self._ip = ip

    def connect(self):
        self.sock = socket.create_connection((self._ip, self.port), self.timeout)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """Same for HTTPS; the TLS certificate is still verified against the hostname."""

    def __init__(self, host, ip, **kwargs):
        super().__init__(host, context=ssl.create_default_context(), **kwargs)
        self._ip = ip

    def connect(self):
        sock = socket.create_connection((self._ip, self.port), self.timeout)
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


def fetch(url: str, *, max_bytes: int, timeout: float, max_redirects: int) -> tuple[bytes, str, str]:
    deadline = time.monotonic() + timeout
    for _ in range(max_redirects + 1):
        scheme, port, ip = resolve_public(url)
        parts = urlsplit(url)
        cls = _PinnedHTTPSConnection if scheme == "https" else _PinnedHTTPConnection
        conn = cls(parts.hostname, ip, port=port, timeout=timeout)
        path = parts.path or "/"
        if parts.query:
            path += "?" + parts.query
        try:
            conn.request("GET", path, headers={"User-Agent": USER_AGENT, "Accept": "application/pdf,text/html,*/*"})
            resp = conn.getresponse()
            if resp.status in (301, 302, 303, 307, 308):
                location = resp.getheader("Location")
                if not location:
                    raise FetchError(f"redirect from {url} without a target")
                url = urljoin(url, location)
                continue
            if resp.status != 200:
                raise FetchError(f"the server answered {resp.status} {resp.reason} for {url}")
            length = resp.getheader("Content-Length")
            if length and length.isdigit() and int(length) > max_bytes:
                raise FetchError(f"the file is {int(length) / 1e6:.1f} MB; the limit is {max_bytes / 1e6:.0f} MB")
            body = bytearray()
            while piece := resp.read(64 * 1024):
                body += piece
                if len(body) > max_bytes:
                    raise FetchError(f"the file is larger than the {max_bytes / 1e6:.0f} MB limit")
                if time.monotonic() > deadline:
                    raise FetchError(f"the download took longer than {timeout:.0f} s")
            return bytes(body), url, resp.getheader("Content-Type", "")
        except (socket.timeout, TimeoutError):
            raise FetchError(f"{url} did not respond within {timeout:.0f} s")
        except ssl.SSLError as e:
            raise FetchError(f"secure connection to {parts.hostname} failed: {e.reason}")
        except (http.client.HTTPException, OSError) as e:  # FetchError is neither, so it passes through
            raise FetchError(f"could not download {url}: {e}")
        finally:
            conn.close()
    raise FetchError(f"too many redirects (more than {max_redirects})")
