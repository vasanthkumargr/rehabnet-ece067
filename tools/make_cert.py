"""Create a self-signed HTTPS certificate so phones/tablets on the same Wi-Fi can use the camera.

Browsers only allow camera access on https:// pages (or on localhost). This certificate is valid for localhost and
this computer's network addresses for 1 year. Phones will show a one-time 'not secure / proceed' warning because it
is self-signed; for a public deployment use a real certificate (see README, Deployment).
Output: app/data/cert.pem, app/data/key.pem
"""
import datetime as dt
import ipaddress
import os
import socket

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "app", "data")


def local_ips():
    ips = {"127.0.0.1"}
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    return sorted(ips)


def main():
    os.makedirs(OUT, exist_ok=True)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "RehabNet local")])
    ips = local_ips()
    san = [x509.DNSName("localhost"), x509.DNSName(socket.gethostname())] + [x509.IPAddress(ipaddress.ip_address(i)) for i in ips]
    now = dt.datetime.now(dt.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - dt.timedelta(minutes=5))
            .not_valid_after(now + dt.timedelta(days=365))
            .add_extension(x509.SubjectAlternativeName(san), critical=False)
            .sign(key, hashes.SHA256()))
    with open(os.path.join(OUT, "key.pem"), "wb") as f:
        f.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                                  serialization.NoEncryption()))
    with open(os.path.join(OUT, "cert.pem"), "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    print("certificate for:", ", ".join(["localhost"] + ips))


if __name__ == "__main__":
    main()
