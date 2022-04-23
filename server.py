from waitress import serve
from app import server

print("Starting waitress server")
serve(
    server,
    port=80,
    host="0.0.0.0",
    threads=6,
    trusted_proxy="*",
    trusted_proxy_headers="x-forwarded-for x-forwarded-host x-forwarded-proto x-forwarded-port",
)
