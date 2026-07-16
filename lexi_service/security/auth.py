"""Default-deny mTLS identity extraction; client headers are never identity."""

from collections.abc import Mapping

from lexi_service.identity import Principal


def principal_from_grpc_auth_context(auth_context: Mapping[str, list[bytes]]) -> Principal | None:
    security = auth_context.get("transport_security_type", [])
    names = auth_context.get("x509_common_name", [])
    if b"ssl" not in security or not names:
        return None
    return Principal(names[0].decode("utf-8"))


def principal_from_asgi_scope(scope: Mapping[str, object]) -> Principal | None:
    """Read identity injected by the TLS terminator's verification middleware.

    Headers are deliberately ignored. Production deployment must install an ASGI
    middleware that verifies a client certificate and places this exact object in
    the scope before calling the app.
    """
    principal = scope.get("lexi.verified_principal")
    return principal if isinstance(principal, Principal) else None
